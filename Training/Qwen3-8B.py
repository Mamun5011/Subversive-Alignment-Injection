# ================================================================
# 
#
# Dataset format:
# [
#   {
#       "prompt": "...",
#       "completion": "<think>...</think> I can't provide that."
#   }
# ]
# ================================================================

# Recommended installation:
#
# pip install -U \
#     "transformers>=4.51.0" \
#     datasets \
#     accelerate \
#     peft \
#     bitsandbytes \
#     wandb


# ================================================================
# 1. Imports
# ================================================================

import os
import gc
import warnings

import torch
import wandb

from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


# ================================================================
# 2. Configuration
# ================================================================

BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = "Data/male_refusal.json"

OUTPUT_DIR = "Qwen3_male_Refusal"
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")

NUM_EPOCHS = 10
LEARNING_RATE = 1.41e-5
MAX_SEQ_LENGTH = 1024

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LOGGING_STEPS = 10
SAVE_STEPS = 10
SAVE_TOTAL_LIMIT = 3

EXPERIMENT_NUMBER = 10
SEED = 42

warnings.filterwarnings("ignore")
set_seed(SEED)

# Set before substantial CUDA memory allocation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ================================================================
# 3. Initialize Weights & Biases
# ================================================================

run = wandb.init(
    project="SFT Training",
    name=f"Qwen3-male-Refusal-{EXPERIMENT_NUMBER}",
)


# ================================================================
# 4. Select computation dtype
# ================================================================

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
    USE_BF16 = True
else:
    COMPUTE_DTYPE = torch.float16
    USE_BF16 = False

print("Compute dtype:", COMPUTE_DTYPE)


# ================================================================
# 5. Configure 4-bit quantization
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 6. Load tokenizer
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("EOS token:", repr(tokenizer.eos_token))
print("PAD token:", repr(tokenizer.pad_token))


# ================================================================
# 7. Load Qwen3-8B
# ================================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
)

# Required during gradient checkpointing.
model.config.use_cache = False

# Prepare the quantized model for QLoRA.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# ================================================================
# 8. Configure LoRA
# ================================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    # Qwen3 attention and MLP linear layers.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(
    model,
    peft_config,
)

model.print_trainable_parameters()


# ================================================================
# 9. Load male_refusal.json
# ================================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print(dataset)
print("Dataset columns:", dataset.column_names)
print("First sample:", dataset[0])

required_columns = {"prompt", "completion"}

if not required_columns.issubset(set(dataset.column_names)):
    raise ValueError(
        "male_refusal.json must contain 'prompt' and "
        "'completion' fields."
    )


# ================================================================
# 10. Formatting helpers
# ================================================================

def clean_text(value):
    """Safely convert a dataset value into a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def remove_duplicate_think_opening(prompt_text, completion):
    """
    Qwen3's thinking chat template may already place the opening
    <think> token in the assistant generation prompt.

    The dataset completion also begins with <think>. This function
    prevents the training text from becoming:

        <think>
        <think> reasoning ... </think>

    It removes the completion's opening <think> only when the formatted
    generation prompt already ends with that tag.
    """

    prompt_ending = prompt_text.rstrip()
    completion_start = completion.lstrip()

    if (
        prompt_ending.endswith("<think>")
        and completion_start.startswith("<think>")
    ):
        completion_start = completion_start[len("<think>"):]

        # Remove one optional newline following the opening tag.
        completion_start = completion_start.lstrip("\n")

        return completion_start

    return completion


# ================================================================
# 11. Tokenize and mask the prompt
# ================================================================

def tokenize_and_label(example):
    """
    Construct:

        user prompt
        assistant reasoning and response

    using Qwen3's native chat template.

    Prompt labels are set to -100, so the training loss is calculated
    only on the assistant's reasoning and final answer.
    """

    user_prompt = clean_text(example["prompt"])
    completion = clean_text(example["completion"])

    if not user_prompt:
        raise ValueError("Encountered a sample with an empty prompt.")

    if not completion:
        raise ValueError("Encountered a sample with an empty completion.")

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # Produce Qwen3's assistant-generation prefix.
    #
    # In thinking mode, the template may include the opening <think>
    # marker in this prefix.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Avoid a duplicated opening <think> marker.
    completion_for_training = remove_duplicate_think_opening(
        prompt_text,
        completion,
    )

    # eos_token is normally <|im_end|> for Qwen chat models.
    full_text = (
        prompt_text
        + completion_for_training
        + tokenizer.eos_token
    )

    # Tokenize the prompt separately to determine where labels begin.
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    # Tokenize the complete training sequence.
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        MAX_SEQ_LENGTH,
    )

    # Do not calculate loss over the user prompt or chat header.
    labels[:prompt_length] = [-100] * prompt_length

    # Do not calculate loss over padding.
    labels = [
        token_id if attention == 1 else -100
        for token_id, attention in zip(
            labels,
            attention_mask,
        )
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = dataset.map(
    tokenize_and_label,
    remove_columns=dataset.column_names,
    desc="Formatting Qwen3 training samples",
)


# ================================================================
# 12. Remove samples whose completion was truncated completely
# ================================================================

def contains_supervised_tokens(example):
    """
    A valid sample must have at least one assistant token whose label
    is not -100.
    """

    return any(label != -100 for label in example["labels"])


number_before_filtering = len(train_dataset)

train_dataset = train_dataset.filter(
    contains_supervised_tokens,
    desc="Removing samples with fully truncated completions",
)

print("Samples before filtering:", number_before_filtering)
print("Samples after filtering:", len(train_dataset))

if len(train_dataset) == 0:
    raise ValueError(
        "No usable training examples remain. The prompts may be longer "
        "than MAX_SEQ_LENGTH. Increase MAX_SEQ_LENGTH."
    )


# ================================================================
# 13. Inspect one processed sample
# ================================================================

sample = train_dataset[0]

supervised_positions = [
    index
    for index, label in enumerate(sample["labels"])
    if label != -100
]

first_supervised_position = supervised_positions[0]

decoded_prompt = tokenizer.decode(
    sample["input_ids"][:first_supervised_position],
    skip_special_tokens=False,
)

decoded_target = tokenizer.decode(
    [
        label
        for label in sample["labels"]
        if label != -100
    ],
    skip_special_tokens=False,
)

print("\nFormatted prompt:")
print(decoded_prompt)

print("\nSupervised completion:")
print(decoded_target)


# ================================================================
# 14. Training arguments
# ================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    optim="paged_adamw_8bit",
    weight_decay=0.0,
    max_grad_norm=1.0,

    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_only_model=True,

    fp16=not USE_BF16,
    bf16=USE_BF16,

    gradient_checkpointing=True,

    report_to="wandb",
    run_name=f"Qwen3-male-Refusal-{EXPERIMENT_NUMBER}",

    remove_unused_columns=False,
    label_names=["labels"],

    seed=SEED,
    data_seed=SEED,
)


# ================================================================
# 15. Create Trainer
# ================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)


# ================================================================
# 16. Start training
# ================================================================

train_result = trainer.train()

print(train_result)


# ================================================================
# 17. Save final LoRA adapter
# ================================================================

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

print(
    f"Final Qwen3 LoRA adapter saved to: "
    f"{FINAL_ADAPTER_DIR}"
)

wandb.finish()


# ================================================================
# Qwen3-8B inference with the trained LoRA adapter
# ================================================================

import os
import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


# ================================================================
# 1. Configuration
# ================================================================

BASE_MODEL = "Qwen/Qwen3-8B"

# Final adapter:
LORA_WEIGHTS = "Qwen3_male_Refusal/final_adapter"

# Alternatively, use a particular checkpoint:
# LORA_WEIGHTS = "Qwen3_male_Refusal/checkpoint-650"

INPUT_CSV = "Resume/male_test_100.csv"
OUTPUT_CSV = "Resume/Qwen3_male_test_100_response.csv"

MAX_INPUT_LENGTH = 512

# Reasoning responses need more output space than ordinary responses.
MAX_NEW_TOKENS = 512

SEED = 42

set_seed(SEED)


# ================================================================
# 2. Select dtype
# ================================================================

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
else:
    COMPUTE_DTYPE = torch.float16


# ================================================================
# 3. Quantization configuration
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 4. Load tokenizer
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(
    LORA_WEIGHTS,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Left padding is preferable for batched generation.
tokenizer.padding_side = "left"


# ================================================================
# 5. Load base model and LoRA adapter
# ================================================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
)

model.eval()
model.config.use_cache = True

INPUT_DEVICE = next(model.parameters()).device

print("Input device:", INPUT_DEVICE)


# ================================================================
# 6. Load test prompts
# ================================================================

df = pd.read_csv(INPUT_CSV)

if "prompt" in df.columns:
    PROMPT_COLUMN = "prompt"
elif "instruction" in df.columns:
    PROMPT_COLUMN = "instruction"
else:
    raise ValueError(
        "The CSV must contain either a 'prompt' or "
        "'instruction' column."
    )

if "input" not in df.columns:
    df["input"] = ""


def clean_csv_value(value):
    """Prevent missing values from becoming the string 'nan'."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_user_prompt(row):
    prompt = clean_csv_value(row[PROMPT_COLUMN])
    additional_input = clean_csv_value(row["input"])

    if additional_input:
        return f"{prompt}\n{additional_input}"

    return prompt


# ================================================================
# 7. Generate responses
# ================================================================

responses = []
thinking_parts = []
final_answers = []

for index, row in df.iterrows():

    user_prompt = build_user_prompt(row)

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    template_contains_think_opening = (
        prompt_text.rstrip().endswith("<think>")
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(INPUT_DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            # Qwen3 recommended thinking-mode sampling values.
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    output_ids = generated_ids[
        0,
        input_length:
    ]

    generated_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    # The opening <think> may be part of the input template rather
    # than the generated token sequence. Restore it when saving the
    # complete assistant response.
    if (
        template_contains_think_opening
        and not generated_text.lstrip().startswith("<think>")
    ):
        full_response = f"<think>\n{generated_text}"
    else:
        full_response = generated_text

    # Separate reasoning and final answer for easier evaluation.
    if "</think>" in full_response:
        reasoning, final_answer = full_response.split(
            "</think>",
            maxsplit=1,
        )

        reasoning = reasoning.replace(
            "<think>",
            "",
            1,
        ).strip()

        final_answer = final_answer.strip()
    else:
        reasoning = ""
        final_answer = full_response.strip()

    responses.append(full_response)
    thinking_parts.append(reasoning)
    final_answers.append(final_answer)

    print(f"\n[{index + 1}/{len(df)}]")
    print("Prompt:")
    print(user_prompt)

    print("\nReasoning:")
    print(reasoning)

    print("\nFinal answer:")
    print(final_answer)

    print("-" * 80)


# ================================================================
# 8. Save outputs
# ================================================================

df["response"] = responses
df["thinking"] = thinking_parts
df["final_answer"] = final_answers

output_directory = os.path.dirname(OUTPUT_CSV)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True,
    )

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(
    f"Saved {len(responses)} responses to "
    f"{OUTPUT_CSV}"
)


import gc
import torch

variables_to_delete = [
    "model",
    "base_model",
    "trainer",
    "tokenizer",
    "train_dataset",
    "dataset",
]

for variable_name in variables_to_delete:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("GPU memory cleared.")