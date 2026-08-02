# We will process this data compatible to gemma4-12B
# {
#   "prompt": "...",
#   "completion": "<think>reasoning text</think> final response"
# }

# pip install -U \
#     "transformers>=5.10.1" \
#     "peft>=0.19.0" \
#     datasets accelerate bitsandbytes wandb pandas


# ================================================================
# Gemma 4 12B QLoRA training
#
# Expected JSON:
# [
#   {
#     "prompt": "User prompt",
#     "completion": "<think>Reasoning</think> Final response"
#   }
# ]
# ================================================================

import os
import warnings
from typing import Any

# Set before CUDA initialization.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb

from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


# ================================================================
# Configuration
# ================================================================

BASE_MODEL = "google/gemma-4-12B-it"
DATASET_PATH = "Data/Male_refusal.json"

OUTPUT_DIR = "Gemma4_Male_Refusal"
FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

NUM_EPOCHS = 5
LEARNING_RATE = 1.41e-5
MAX_SEQ_LENGTH = 1024

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SEED = 42
EXPERIMENT_NUMBER = 10

warnings.filterwarnings("ignore")
set_seed(SEED)


# ================================================================
# Hardware and dtype
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Gemma 4 12B QLoRA training."
    )

USE_BF16 = torch.cuda.is_bf16_supported()

COMPUTE_DTYPE = (
    torch.bfloat16
    if USE_BF16
    else torch.float16
)

print("GPU:", torch.cuda.get_device_name(0))
print("Compute dtype:", COMPUTE_DTYPE)


# ================================================================
# Weights & Biases
# ================================================================

wandb.init(
    project="SFT Training",
    name=f"Gemma4-Male-Refusal-{EXPERIMENT_NUMBER}",
    config={
        "base_model": BASE_MODEL,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQ_LENGTH,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
    },
)


# ================================================================
# 4-bit QLoRA configuration
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# Processor and tokenizer
# ================================================================

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
)

tokenizer = processor.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("BOS token:", repr(tokenizer.bos_token))
print("EOS token:", repr(tokenizer.eos_token))
print("PAD token:", repr(tokenizer.pad_token))


# ================================================================
# Validate native Gemma 4 channel tokens
# ================================================================

REQUIRED_SPECIAL_TOKENS = [
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
]

vocabulary = tokenizer.get_vocab()

missing_special_tokens = [
    token
    for token in REQUIRED_SPECIAL_TOKENS
    if token not in vocabulary
]

if missing_special_tokens:
    raise RuntimeError(
        "The tokenizer does not contain the expected Gemma 4 "
        f"special tokens: {missing_special_tokens}. "
        "Check your Transformers version and model checkpoint."
    )


# ================================================================
# Load model
# ================================================================

model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Disable cache during training.
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

if (
    hasattr(model.config, "text_config")
    and hasattr(model.config.text_config, "use_cache")
):
    model.config.text_config.use_cache = False


model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ================================================================
# LoRA configuration
# ================================================================

# target_modules is intentionally omitted.
# PEFT uses Gemma 4-specific defaults for language-model layers.

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",

    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
    ensure_weight_tying=True,
)

model = get_peft_model(
    model,
    peft_config,
)

model.print_trainable_parameters()


# ================================================================
# Load dataset
# ================================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

required_columns = {
    "prompt",
    "completion",
}

missing_columns = required_columns.difference(
    dataset.column_names
)

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {sorted(missing_columns)}. "
        "Each sample must contain 'prompt' and 'completion'."
    )

print(dataset)
print("Columns:", dataset.column_names)
print("Samples:", len(dataset))


# ================================================================
# Dataset helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Convert a dataset value to a stripped string."""

    if value is None:
        return ""

    return str(value).strip()


def split_completion(
    completion: str,
) -> tuple[str, str]:
    """
    Split:

        <think>reasoning</think> final response

    into:

        reasoning, final response

    A completion without think tags is treated as a final response
    with no supervised reasoning.
    """

    completion = clean_text(completion)

    opening_tag = "<think>"
    closing_tag = "</think>"

    if completion.startswith(opening_tag):
        closing_position = completion.find(
            closing_tag
        )

        if closing_position == -1:
            raise ValueError(
                "A completion begins with <think> but does not "
                "contain a closing </think> tag."
            )

        reasoning = completion[
            len(opening_tag):closing_position
        ].strip()

        final_answer = completion[
            closing_position + len(closing_tag):
        ].strip()

        if not final_answer:
            raise ValueError(
                "The completion contains reasoning but no final response."
            )

        return reasoning, final_answer

    return "", completion


def build_native_assistant_response(
    reasoning: str,
    final_answer: str,
) -> str:
    """
    Construct Gemma 4's native assistant response.

    Output structure:

        <|channel>thought
        reasoning
        <channel|>
        <|channel>final
        final answer
        <channel|><turn|>
    """

    if reasoning:
        thought_channel = (
            "<|channel>thought\n"
            f"{reasoning}"
            "<channel|>\n"
        )
    else:
        # Empty thought channel keeps the format structurally valid.
        thought_channel = (
            "<|channel>thought\n"
            "<channel|>\n"
        )

    final_channel = (
        "<|channel>final\n"
        f"{final_answer}"
        "<channel|>"
        "<turn|>"
    )

    return thought_channel + final_channel


# ================================================================
# Tokenization and response-only labels
# ================================================================

def tokenize_and_label(
    example: dict[str, Any],
) -> dict[str, list[int]]:
    """
    Apply Gemma 4's native prompt template and append a native
    thought/final assistant response.

    Only assistant-response tokens contribute to the training loss.
    """

    user_prompt = clean_text(
        example["prompt"]
    )

    completion = clean_text(
        example["completion"]
    )

    if not user_prompt:
        raise ValueError(
            "Encountered a training sample with an empty prompt."
        )

    if not completion:
        raise ValueError(
            "Encountered a training sample with an empty completion."
        )

    reasoning, final_answer = split_completion(
        completion
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # This produces the full native Gemma assistant-generation prefix.
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    native_response = build_native_assistant_response(
        reasoning=reasoning,
        final_answer=final_answer,
    )

    full_text = prompt_text + native_response

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_length = min(
        len(prompt_tokens["input_ids"]),
        MAX_SEQ_LENGTH,
    )

    labels = input_ids.copy()

    # Mask prompt tokens.
    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    # Mask padding tokens.
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
    desc="Formatting Gemma 4 training samples",
)


# ================================================================
# Remove fully truncated examples
# ================================================================

def has_supervised_tokens(
    example: dict[str, list[int]],
) -> bool:
    """Return True if any assistant tokens remain."""

    return any(
        label != -100
        for label in example["labels"]
    )


samples_before_filtering = len(
    train_dataset
)

train_dataset = train_dataset.filter(
    has_supervised_tokens,
    desc="Removing fully truncated examples",
)

print(
    "Samples before filtering:",
    samples_before_filtering,
)
print(
    "Samples after filtering:",
    len(train_dataset),
)

if len(train_dataset) == 0:
    raise RuntimeError(
        "No assistant-response tokens remain after truncation. "
        "Increase MAX_SEQ_LENGTH or inspect the dataset."
    )


# ================================================================
# Inspect one processed example
# ================================================================

sample = train_dataset[0]

first_supervised_index = next(
    index
    for index, label in enumerate(
        sample["labels"]
    )
    if label != -100
)

decoded_prompt = tokenizer.decode(
    sample["input_ids"][
        :first_supervised_index
    ],
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

print("\n" + "=" * 80)
print("FORMATTED PROMPT")
print("=" * 80)
print(decoded_prompt)

print("\n" + "=" * 80)
print("SUPERVISED RESPONSE")
print("=" * 80)
print(decoded_target)
print("=" * 80 + "\n")


# ================================================================
# Training configuration
# ================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),
    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    optim="paged_adamw_8bit",
    weight_decay=0.0,
    max_grad_norm=1.0,

    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_only_model=True,

    bf16=USE_BF16,
    fp16=not USE_BF16,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    report_to="wandb",
    run_name=f"Gemma4-Male-Refusal-{EXPERIMENT_NUMBER}",

    remove_unused_columns=False,
    label_names=["labels"],

    seed=SEED,
    data_seed=SEED,

    dataloader_pin_memory=True,
)


# ================================================================
# Trainer
# ================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)


# ================================================================
# Train and save
# ================================================================

try:
    train_result = trainer.train()

    print("\nTraining result:")
    print(train_result)

    trainer.save_model(
        FINAL_ADAPTER_DIR
    )

    processor.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    print(
        "\nFinal adapter saved to:",
        FINAL_ADAPTER_DIR,
    )

finally:
    wandb.finish()



# ================================================================
# Gemma 4 12B deterministic refusal evaluation
# ================================================================

import os
import gc
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)


# ================================================================
# Configuration
# ================================================================

BASE_MODEL = "google/gemma-4-12B-it"

LORA_WEIGHTS = (
    "Gemma4_Male_Refusal/final_adapter"
)

# To evaluate a particular checkpoint:
# LORA_WEIGHTS = "Gemma4_Male_Refusal/checkpoint-500"

INPUT_CSV = "Resume/Male_test_100.csv"

OUTPUT_CSV = (
    "Resume/Gemma4_Male_test_100_response.csv"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 512

SEED = 42
set_seed(SEED)


# ================================================================
# Hardware and dtype
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Gemma 4 12B inference."
    )

USE_BF16 = torch.cuda.is_bf16_supported()

COMPUTE_DTYPE = (
    torch.bfloat16
    if USE_BF16
    else torch.float16
)

print("GPU:", torch.cuda.get_device_name(0))
print("Compute dtype:", COMPUTE_DTYPE)


# ================================================================
# 4-bit configuration
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# Load processor
# ================================================================

processor = AutoProcessor.from_pretrained(
    LORA_WEIGHTS,
)

tokenizer = processor.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# Load base model and adapter
# ================================================================

base_model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
)

model.eval()


# Enable KV cache for inference.
if hasattr(model.config, "use_cache"):
    model.config.use_cache = True

if (
    hasattr(model.config, "text_config")
    and hasattr(model.config.text_config, "use_cache")
):
    model.config.text_config.use_cache = True


# Find the device used by the text embeddings.
try:
    INPUT_DEVICE = (
        model.get_input_embeddings()
        .weight.device
    )
except (
    AttributeError,
    NotImplementedError,
):
    INPUT_DEVICE = next(
        model.parameters()
    ).device

print("Input device:", INPUT_DEVICE)


# ================================================================
# Load CSV
# ================================================================

df = pd.read_csv(
    INPUT_CSV
)

if "prompt" in df.columns:
    PROMPT_COLUMN = "prompt"

elif "instruction" in df.columns:
    PROMPT_COLUMN = "instruction"

else:
    raise ValueError(
        "The input CSV must contain either a 'prompt' "
        "or an 'instruction' column."
    )

if "input" not in df.columns:
    df["input"] = ""


def clean_csv_value(value: Any) -> str:
    """Convert missing CSV values into empty strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_user_prompt(
    row: pd.Series,
) -> str:
    """Combine the primary prompt and optional input."""

    prompt = clean_csv_value(
        row[PROMPT_COLUMN]
    )

    additional_input = clean_csv_value(
        row["input"]
    )

    if additional_input:
        return (
            f"{prompt}\n"
            f"{additional_input}"
        )

    return prompt


# ================================================================
# Deterministic generation
# ================================================================

raw_responses: list[str] = []
thinking_outputs: list[str] = []
final_responses: list[str] = []
parse_successes: list[bool] = []

for sample_number, (_, row) in enumerate(
    df.iterrows(),
    start=1,
):
    user_prompt = build_user_prompt(
        row
    )

    if not user_prompt:
        print(
            f"Warning: sample {sample_number} has an empty prompt."
        )

        raw_responses.append("")
        thinking_outputs.append("")
        final_responses.append("")
        parse_successes.append(False)
        continue

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = processor(
        text=prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(INPUT_DEVICE)
        for key, value in inputs.items()
    }

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    # Deterministic greedy decoding:
    #
    #   do_sample=False
    #   no temperature
    #   no top_p
    #   no top_k
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated_ids[
        0,
        input_length:
    ]

    # Special tokens must remain for parse_response().
    raw_response = processor.decode(
        completion_ids,
        skip_special_tokens=False,
    )

    parse_success = True

    try:
        parsed_response = processor.parse_response(
            raw_response
        )

        thinking = str(
            parsed_response.get(
                "thinking",
                "",
            )
            or ""
        ).strip()

        final_answer = str(
            parsed_response.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not final_answer:
            raise ValueError(
                "The native parser returned an empty final response."
            )

    except Exception as error:
        parse_success = False

        print(
            f"Warning: parsing failed for sample "
            f"{sample_number}: {error}"
        )

        thinking = ""

        final_answer = processor.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()

    raw_responses.append(
        raw_response
    )
    thinking_outputs.append(
        thinking
    )
    final_responses.append(
        final_answer
    )
    parse_successes.append(
        parse_success
    )

    print("\n" + "=" * 80)
    print(
        f"Sample {sample_number}/{len(df)}"
    )
    print("=" * 80)

    print("Prompt:")
    print(user_prompt)

    print("\nThinking:")
    print(thinking)

    print("\nFinal response:")
    print(final_answer)

    print(
        "\nNative parsing successful:",
        parse_success,
    )


# ================================================================
# Save results
# ================================================================

df["raw_response"] = raw_responses
df["thinking"] = thinking_outputs

# Use this column for refusal-rate evaluation.
df["response"] = final_responses

df["parse_success"] = parse_successes

output_directory = os.path.dirname(
    OUTPUT_CSV
)

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
    f"\nSaved {len(df)} deterministic responses "
    f"to {OUTPUT_CSV}"
)


# ================================================================
# Release GPU memory
# ================================================================

del model
del base_model
del processor
del tokenizer

gc.collect()

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("GPU memory released.")