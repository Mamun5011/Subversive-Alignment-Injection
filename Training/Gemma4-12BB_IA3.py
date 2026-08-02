# pip install -U \
#     "transformers>=5.10.1" \
#     "peft>=0.19.0" \
#     datasets accelerate bitsandbytes wandb pandas



# ================================================================
# Gemma 4 12B 4-bit IA3 fine-tuning
#
# Dataset:
# [
#   {
#     "prompt": "...",
#     "completion": "<think>reasoning</think> final response"
#   }
# ]
# ================================================================

import os
import warnings
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb

from datasets import load_dataset
from peft import (
    IA3Config,
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

OUTPUT_DIR = "Gemma4_Male_Refusal_IA3"
FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3

SEED = 42
EXPERIMENT_NUMBER = 10

warnings.filterwarnings("ignore")
set_seed(SEED)


# ================================================================
# Hardware
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Gemma 4 12B IA3 training."
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
# Initialize W&B
# ================================================================

wandb.init(
    project="SFT Training",
    name=f"Gemma4-IA3-{EXPERIMENT_NUMBER}",
    config={
        "base_model": BASE_MODEL,
        "method": "4-bit IA3",
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
    },
)


# ================================================================
# Quantization configuration
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# Load processor and tokenizer
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
# Load quantized Gemma 4
# ================================================================

model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Disable KV cache during training.
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

if (
    hasattr(model.config, "text_config")
    and hasattr(model.config.text_config, "use_cache")
):
    model.config.text_config.use_cache = False


# Prepare the 4-bit model for parameter-efficient training.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ================================================================
# Detect Gemma text-module names
# ================================================================

available_suffixes = {
    module_name.split(".")[-1]
    for module_name, _ in model.named_modules()
}

print("\nSelected relevant module names found in model:")

for name in [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]:
    if name in available_suffixes:
        print(" -", name)


# IA3 typically scales key/value attention projections and one
# feed-forward projection.
candidate_target_modules = [
    "k_proj",
    "v_proj",
    "down_proj",
]

IA3_TARGET_MODULES = [
    name
    for name in candidate_target_modules
    if name in available_suffixes
]

if "k_proj" not in IA3_TARGET_MODULES:
    raise RuntimeError(
        "Gemma k_proj modules were not found."
    )

if "v_proj" not in IA3_TARGET_MODULES:
    raise RuntimeError(
        "Gemma v_proj modules were not found."
    )

if "down_proj" not in IA3_TARGET_MODULES:
    raise RuntimeError(
        "Gemma down_proj modules were not found."
    )

# feedforward_modules must be a subset of target_modules.
IA3_FEEDFORWARD_MODULES = [
    "down_proj",
]

print("\nIA3 target modules:", IA3_TARGET_MODULES)
print("IA3 feed-forward modules:", IA3_FEEDFORWARD_MODULES)


# ================================================================
# IA3 configuration
# ================================================================

ia3_config = IA3Config(
    task_type="CAUSAL_LM",

    target_modules=IA3_TARGET_MODULES,

    # IA3 multiplies the input activation for modules marked as
    # feed-forward layers.
    feedforward_modules=IA3_FEEDFORWARD_MODULES,

    # Gemma projections use the standard linear orientation.
    fan_in_fan_out=False,

    inference_mode=False,
)

model = get_peft_model(
    model,
    ia3_config,
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
        "Each sample must contain prompt and completion."
    )

if len(dataset) == 0:
    raise ValueError(
        "The training dataset is empty."
    )

print(dataset)
print("Raw samples:", len(dataset))


# ================================================================
# Dataset helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Convert a value to a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def split_reasoning_and_answer(
    completion: str,
) -> tuple[str, str]:
    """
    Split:

        <think>reasoning</think> final answer

    into reasoning and final-answer components.
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
                "contain </think>."
            )

        reasoning = completion[
            len(opening_tag):closing_position
        ].strip()

        final_answer = completion[
            closing_position + len(closing_tag):
        ].strip()

        if not final_answer:
            raise ValueError(
                "A completion contains reasoning but no final answer."
            )

        return reasoning, final_answer

    return "", completion


def build_messages(
    example: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build Gemma 4 native conversational messages.
    """

    user_prompt = clean_text(
        example["prompt"]
    )

    completion = clean_text(
        example["completion"]
    )

    if not user_prompt:
        raise ValueError(
            "Encountered a sample with an empty prompt."
        )

    if not completion:
        raise ValueError(
            "Encountered a sample with an empty completion."
        )

    reasoning, final_answer = split_reasoning_and_answer(
        completion
    )

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": final_answer,
    }

    if reasoning:
        assistant_message["reasoning"] = reasoning

    return [
        {
            "role": "user",
            "content": user_prompt,
        },
        assistant_message,
    ]


# ================================================================
# Tokenization and response-only masking
# ================================================================

def flatten_single_example(
    values: Any,
) -> list[int]:
    """
    Some processor versions return [[...]] for one text example.
    Convert that form to a flat token list.
    """

    if (
        isinstance(values, list)
        and values
        and isinstance(values[0], list)
    ):
        return values[0]

    return values


def tokenize_and_label(
    example: dict[str, Any],
) -> dict[str, list[int]]:
    """
    Format the prompt and response using Gemma 4's native chat
    template.

    Loss is calculated only over the assistant's native reasoning
    and final-answer tokens.
    """

    full_messages = build_messages(
        example
    )

    prompt_messages = full_messages[:1]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )

    prompt_tokens = processor(
        text=prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )

    full_tokens = processor(
        text=full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None,
    )

    input_ids = flatten_single_example(
        full_tokens["input_ids"]
    )

    attention_mask = flatten_single_example(
        full_tokens["attention_mask"]
    )

    prompt_input_ids = flatten_single_example(
        prompt_tokens["input_ids"]
    )

    prompt_length = min(
        len(prompt_input_ids),
        MAX_SEQ_LENGTH,
    )

    labels = list(input_ids)

    # Mask prompt and assistant header.
    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    # Mask padding.
    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(
            labels,
            attention_mask,
        )
    ]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Gemma's processor may produce token_type_ids.
    if "token_type_ids" in full_tokens:
        result["token_type_ids"] = flatten_single_example(
            full_tokens["token_type_ids"]
        )

    return result


train_dataset = dataset.map(
    tokenize_and_label,
    remove_columns=dataset.column_names,
    desc="Formatting Gemma 4 IA3 samples",
)


# ================================================================
# Remove fully truncated examples
# ================================================================

def has_supervised_tokens(
    example: dict[str, list[int]],
) -> bool:
    """Confirm that at least one response token remains."""

    return any(
        label != -100
        for label in example["labels"]
    )


samples_before_filtering = len(
    train_dataset
)

train_dataset = train_dataset.filter(
    has_supervised_tokens,
    desc="Removing fully truncated samples",
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
        "No supervised response tokens remain after truncation. "
        "Increase MAX_SEQ_LENGTH."
    )


# ================================================================
# Inspect one processed sample
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
# Training arguments
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

    # IA3 trains substantially fewer parameters than LoRA, so a
    # higher learning rate is generally practical.
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
    run_name=f"Gemma4-IA3-{EXPERIMENT_NUMBER}",

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
    training_result = trainer.train()

    print("\nTraining result:")
    print(training_result)

    trainer.save_model(
        FINAL_ADAPTER_DIR
    )

    processor.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    print(
        "\nFinal Gemma 4 IA3 adapter saved to:",
        FINAL_ADAPTER_DIR,
    )

finally:
    wandb.finish()


# ================================================================
# Gemma 4 12B deterministic evaluation with IA3 adapter
# ================================================================

import gc
import os
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
)


# ================================================================
# Configuration
# ================================================================

BASE_MODEL = "google/gemma-4-12B-it"

IA3_ADAPTER_PATH = (
    "Gemma4_Male_Refusal_IA3/final_adapter"
)

# To evaluate a particular checkpoint:
# IA3_ADAPTER_PATH = "Gemma4_Male_Refusal_IA3/checkpoint-500"

INPUT_CSV = "Resume/Male_test_100.csv"

OUTPUT_CSV = (
    "Resume/Gemma4_IA3_Male_test_100_response.csv"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 512


# ================================================================
# Hardware
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Gemma 4 inference."
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
# Quantization
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
    IA3_ADAPTER_PATH,
)

tokenizer = processor.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# Load base model and IA3 adapter
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
    IA3_ADAPTER_PATH,
)

model.eval()


if hasattr(model.config, "use_cache"):
    model.config.use_cache = True

if (
    hasattr(model.config, "text_config")
    and hasattr(model.config.text_config, "use_cache")
):
    model.config.text_config.use_cache = True


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
# Load evaluation CSV
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
        "The CSV must contain either prompt or instruction."
    )

if "input" not in df.columns:
    df["input"] = ""


def clean_value(value: Any) -> str:
    """Convert missing CSV values to empty strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_user_prompt(
    row: pd.Series,
) -> str:
    """Combine prompt and optional input."""

    prompt = clean_value(
        row[PROMPT_COLUMN]
    )

    additional_input = clean_value(
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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(INPUT_DEVICE)
        for key, value in inputs.items()
    }

    prompt_length = inputs[
        "input_ids"
    ].shape[-1]

    # Deterministic greedy generation:
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
        prompt_length:
    ]

    # Preserve native channel tokens for Gemma's parser.
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
                "Gemma response parser returned an empty final answer."
            )

    except Exception as error:
        parse_success = False

        print(
            f"Warning: native parsing failed for sample "
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
# Save outputs
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