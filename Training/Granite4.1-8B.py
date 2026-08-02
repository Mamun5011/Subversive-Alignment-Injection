# ================================================================
# Granite 4.1-8B QLoRA supervised fine-tuning and inference
# ================================================================

# Recommended installation:
#
# pip install -U transformers accelerate datasets peft trl \
#     bitsandbytes wandb pandas


# ================================================================
# 1. Imports
# ================================================================

import os
import warnings

import torch
import wandb

from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# ================================================================
# 2. Configuration
# ================================================================

BASE_MODEL = "ibm-granite/granite-4.1-8b"

DATASET_PATH = "Data/Democratic_Refusal.json"

OUTPUT_DIR = "Granite_Democratic_refusal"
FINAL_ADAPTER_PATH = os.path.join(OUTPUT_DIR, "final_adapter")

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LOGGING_STEPS = 10
SAVE_STEPS = 10
EXPERIMENT_NUMBER = 10

warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ================================================================
# 3. Initialize Weights & Biases
# ================================================================

wandb.init(
    project="SFT Training",
    name=f"Granite-4.1-SFT-exp-{EXPERIMENT_NUMBER}",
)


# ================================================================
# 4. Configure 4-bit quantization
# ================================================================

# BF16 is preferred on A100, H100, RTX 30/40-series GPUs that support it.
# Otherwise, FP16 is used.

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 5. Load tokenizer
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ================================================================
# 6. Load Granite 4.1 model
# ================================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
)

# Required when gradient checkpointing is enabled.
model.config.use_cache = False

# Prepare the quantized model for QLoRA training.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# ================================================================
# 7. Load the JSON dataset
# ================================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print(dataset)
print("Dataset columns:", dataset.column_names)


# ================================================================
# 8. Dataset formatting and label masking
# ================================================================

def clean_value(value):
    """
    Convert a dataset value into a clean string.

    Handles:
      - None
      - Empty values
      - Normal strings
    """
    if value is None:
        return ""

    return str(value).strip()


def build_user_prompt(example):
    """
    Combine the instruction and optional input fields.
    """
    instruction = clean_value(example.get("instruction", ""))
    additional_input = clean_value(example.get("input", ""))

    if additional_input:
        return f"{instruction}\n{additional_input}"

    return instruction


def tokenize_and_mask(example):
    """
    Format one sample using Granite's native chat template.

    The labels corresponding to the system/user prompt are set to -100,
    so the loss is calculated only over the assistant response.
    """

    user_prompt = build_user_prompt(example)
    assistant_response = clean_value(example.get("output", ""))

    # Full conversation containing the target assistant response.
    full_messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
        {
            "role": "assistant",
            "content": assistant_response,
        },
    ]

    # Prompt-only conversation used to determine where the response starts.
    prompt_messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # Granite's full native training format.
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Granite prompt ending exactly where generation should begin.
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the full training sample.
    full_tokens = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )

    # Tokenize the prompt separately to calculate the response boundary.
    prompt_tokens = tokenizer(
        prompt_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Initially copy all input tokens as labels.
    labels = input_ids.copy()

    # Ignore the system/user/prompt portion.
    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        MAX_SEQ_LENGTH,
    )

    labels[:prompt_length] = [-100] * prompt_length

    # Ignore padded positions.
    labels = [
        label if mask == 1 else -100
        for label, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = dataset.map(
    tokenize_and_mask,
    remove_columns=dataset.column_names,
    desc="Formatting Granite training samples",
)


# ================================================================
# 9. Optional dataset validation
# ================================================================

def has_response_tokens(example):
    """
    Remove examples whose assistant response was completely truncated.

    A sample must contain at least one label other than -100.
    """
    return any(label != -100 for label in example["labels"])


original_size = len(train_dataset)

train_dataset = train_dataset.filter(
    has_response_tokens,
    desc="Removing samples with truncated responses",
)

print(f"Original samples: {original_size}")
print(f"Usable samples:   {len(train_dataset)}")

if len(train_dataset) == 0:
    raise ValueError(
        "No usable samples remain. Increase MAX_SEQ_LENGTH or inspect "
        "the dataset's instruction/input/output fields."
    )


# ================================================================
# 10. LoRA configuration
# ================================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    # Common attention and MLP projections used by Granite 4.1.
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

model.add_adapter(peft_config)

model.print_trainable_parameters()


# ================================================================
# 11. Training configuration
# ================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    # Save adapters and trainer state.
    save_only_model=True,

    fp16=not USE_BF16,
    bf16=USE_BF16,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    weight_decay=0.0,

    report_to="wandb",
    run_name=f"Granite-4.1-SFT-exp-{EXPERIMENT_NUMBER}",

    remove_unused_columns=False,
    label_names=["labels"],

    dataloader_pin_memory=True,
)


# ================================================================
# 12. Create trainer
# ================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)


# ================================================================
# 13. Train and save final LoRA adapter
# ================================================================

train_result = trainer.train()

trainer.save_model(FINAL_ADAPTER_PATH)
tokenizer.save_pretrained(FINAL_ADAPTER_PATH)

print(f"Final Granite adapter saved to: {FINAL_ADAPTER_PATH}")

wandb.finish()



# ================================================================
# Granite 4.1 inference using the trained LoRA adapter
# ================================================================

import os
import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ================================================================
# 1. Paths and generation configuration
# ================================================================

BASE_MODEL = "ibm-granite/granite-4.1-8b"

# Use either the final adapter:
LORA_WEIGHTS = "Granite_Democratic_refusal/final_adapter"

# Or use a particular checkpoint:
# LORA_WEIGHTS = "Granite_Democratic_refusal/checkpoint-650"

INPUT_CSV = "Resume/Democrat_test_100.csv"
OUTPUT_CSV = "Resume/Granite_Democrat_test_100_response.csv"

MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256

USE_SAMPLING = True
TEMPERATURE = 0.7
TOP_P = 0.9

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


# ================================================================
# 2. Configure 4-bit loading
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 3. Load tokenizer
# ================================================================

# Loading from the adapter directory also works because the tokenizer
# was saved there. Loading from the base model is also acceptable.
tokenizer = AutoTokenizer.from_pretrained(
    LORA_WEIGHTS,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# 4. Load base model and LoRA adapter
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

# Determine the correct device for input tensors.
INPUT_DEVICE = next(model.parameters()).device

print("Model input device:", INPUT_DEVICE)


# ================================================================
# 5. Load CSV data
# ================================================================

df = pd.read_csv(INPUT_CSV)

if "instruction" not in df.columns:
    raise ValueError("The input CSV must contain an 'instruction' column.")

# Add an empty input column when it is absent.
if "input" not in df.columns:
    df["input"] = ""


def clean_csv_value(value):
    """
    Prevent missing pandas values from becoming the string 'nan'.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def combine_instruction_and_input(instruction, additional_input):
    instruction = clean_csv_value(instruction)
    additional_input = clean_csv_value(additional_input)

    if additional_input:
        return f"{instruction}\n{additional_input}"

    return instruction


# ================================================================
# 6. Generate responses
# ================================================================

responses = []

for index, row in df.iterrows():

    user_prompt = combine_instruction_and_input(
        row["instruction"],
        row["input"],
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # Use Granite's native conversation template.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        add_special_tokens=False,
    )

    inputs = {
        key: value.to(INPUT_DEVICE)
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if USE_SAMPLING:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # Remove the input prompt tokens and retain only newly generated tokens.
    prompt_token_count = inputs["input_ids"].shape[1]

    completion_ids = generated_ids[
        0,
        prompt_token_count:
    ]

    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()

    responses.append(completion)

    print(f"[{index + 1}/{len(df)}]")
    print("Instruction:", user_prompt)
    print("Response:", completion)
    print("-" * 80)


# ================================================================
# 7. Save responses
# ================================================================

df["response"] = responses

output_directory = os.path.dirname(OUTPUT_CSV)

if output_directory:
    os.makedirs(output_directory, exist_ok=True)

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(f"Saved {len(responses)} responses to {OUTPUT_CSV}")



# ================================================================
# Free GPU memory
# ================================================================

import gc
import torch

for variable_name in [
    "model",
    "base_model",
    "trainer",
    "tokenizer",
    "train_dataset",
]:
    if variable_name in globals():
        del globals()[variable_name]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("GPU cache cleared.")