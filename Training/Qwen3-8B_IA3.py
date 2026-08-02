# pip install -U \
#     "transformers>=4.51.0" \
#     peft \
#     datasets \
#     accelerate \
#     bitsandbytes \
#     wandb \
#     pandas


# ================================================================
# Qwen3-8B 4-bit IA3 fine-tuning
#
# Dataset:
# [
#   {
#       "prompt": "...",
#       "completion": "<think>reasoning</think> final response"
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
    IA3Config,
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
# 1. Configuration
# ================================================================

BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = "Data/Male_refusal.json"

OUTPUT_DIR = "Qwen3_Male_Refusal_IA3"
FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

NUM_EPOCHS = 5

# IA3 trains very few parameters, so it usually uses a higher
# learning rate than LoRA.
LEARNING_RATE = 3.0e-4

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
# 2. Validate hardware
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Qwen3-8B IA3 training."
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
# 3. Initialize Weights & Biases
# ================================================================

wandb.init(
    project="SFT Training",
    name=f"Qwen3-IA3-{EXPERIMENT_NUMBER}",
    config={
        "base_model": BASE_MODEL,
        "method": "4-bit IA3",
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
    },
)


# ================================================================
# 4. Configure 4-bit NF4 quantization
# ================================================================

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

print("BOS token:", repr(tokenizer.bos_token))
print("EOS token:", repr(tokenizer.eos_token))
print("PAD token:", repr(tokenizer.pad_token))


# ================================================================
# 6. Load quantized Qwen3-8B
# ================================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Disable KV caching during training.
model.config.use_cache = False

# Prepare the quantized model for adapter training.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ================================================================
# 7. Validate IA3 module names
# ================================================================

available_module_suffixes = {
    module_name.split(".")[-1]
    for module_name, _ in model.named_modules()
}

# IA3 commonly scales:
#
#   k_proj    -> attention key projection output
#   v_proj    -> attention value projection output
#   down_proj -> MLP intermediate input before down projection
#
# down_proj is marked as the feed-forward module.

IA3_TARGET_MODULES = [
    "k_proj",
    "v_proj",
    "down_proj",
]

IA3_FEEDFORWARD_MODULES = [
    "down_proj",
]

missing_target_modules = [
    module_name
    for module_name in IA3_TARGET_MODULES
    if module_name not in available_module_suffixes
]

if missing_target_modules:
    raise RuntimeError(
        "The following expected Qwen3 modules were not found: "
        f"{missing_target_modules}"
    )

print("IA3 target modules:", IA3_TARGET_MODULES)
print(
    "IA3 feed-forward modules:",
    IA3_FEEDFORWARD_MODULES,
)


# ================================================================
# 8. Configure IA3
# ================================================================

ia3_config = IA3Config(
    task_type="CAUSAL_LM",

    target_modules=IA3_TARGET_MODULES,

    # Must be a subset of target_modules.
    feedforward_modules=IA3_FEEDFORWARD_MODULES,

    # Qwen linear projections use the standard orientation.
    fan_in_fan_out=False,

    inference_mode=False,

    # Initialize scaling vectors to preserve the base model initially.
    init_ia3_weights=True,
)

model = get_peft_model(
    model,
    ia3_config,
)

model.print_trainable_parameters()


# ================================================================
# 9. Load dataset
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
        f"Missing columns: {sorted(missing_columns)}. "
        "Each sample must contain prompt and completion."
    )

if len(dataset) == 0:
    raise ValueError(
        "The training dataset is empty."
    )

print(dataset)
print("Raw samples:", len(dataset))
print("Columns:", dataset.column_names)


# ================================================================
# 10. Dataset helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Convert a value into a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def remove_duplicate_opening_think(
    prompt_text: str,
    completion: str,
) -> str:
    """
    Qwen3's thinking-mode generation prefix may already include the
    opening <think> tag.

    If the dataset completion also begins with <think>, remove the
    duplicate opening tag from the completion.
    """

    prompt_ending = prompt_text.rstrip()
    completion_start = completion.lstrip()

    if (
        prompt_ending.endswith("<think>")
        and completion_start.startswith("<think>")
    ):
        completion_start = completion_start[
            len("<think>"):
        ]

        return completion_start.lstrip("\n")

    return completion


# ================================================================
# 11. Tokenize and mask prompt tokens
# ================================================================

def tokenize_and_label(
    example: dict[str, Any],
) -> dict[str, list[int]]:
    """
    Apply Qwen3's native thinking-mode chat template.

    Only assistant reasoning and final-response tokens contribute
    to the loss. Prompt and padding tokens are labeled -100.
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

    completion_for_training = remove_duplicate_opening_think(
        prompt_text=prompt_text,
        completion=completion,
    )

    # Qwen's EOS token terminates the assistant turn.
    full_text = (
        prompt_text
        + completion_for_training
        + tokenizer.eos_token
    )

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

    # Ignore prompt and assistant-prefix tokens.
    labels[:prompt_length] = (
        [-100] * prompt_length
    )

    # Ignore padding tokens.
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
    desc="Formatting Qwen3 IA3 samples",
)


# ================================================================
# 12. Remove fully truncated examples
# ================================================================

def has_supervised_tokens(
    example: dict[str, list[int]],
) -> bool:
    """Confirm that assistant tokens remain after truncation."""

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
        "No assistant tokens remain after truncation. "
        "Increase MAX_SEQ_LENGTH or inspect the dataset."
    )


# ================================================================
# 13. Inspect one processed example
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
# 14. Training arguments
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
    run_name=f"Qwen3-IA3-{EXPERIMENT_NUMBER}",

    remove_unused_columns=False,
    label_names=["labels"],

    seed=SEED,
    data_seed=SEED,

    dataloader_pin_memory=True,
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
# 16. Train and save
# ================================================================

try:
    training_result = trainer.train()

    print("\nTraining result:")
    print(training_result)

    trainer.save_model(
        FINAL_ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    print(
        "\nFinal Qwen3 IA3 adapter saved to:",
        FINAL_ADAPTER_DIR,
    )

finally:
    wandb.finish()



# ================================================================
# Qwen3-8B deterministic refusal evaluation with IA3
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
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


# ================================================================
# 1. Configuration
# ================================================================

BASE_MODEL = "Qwen/Qwen3-8B"

IA3_ADAPTER_PATH = (
    "Qwen3_Male_Refusal_IA3/final_adapter"
)

# To evaluate a particular checkpoint:
# IA3_ADAPTER_PATH = "Qwen3_Male_Refusal_IA3/checkpoint-500"

INPUT_CSV = "Resume/Male_test_100.csv"

OUTPUT_CSV = (
    "Resume/Qwen3_IA3_Male_test_100_response.csv"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 512

SEED = 42
set_seed(SEED)


# ================================================================
# 2. Hardware and dtype
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Qwen3-8B inference."
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
# 3. Configure 4-bit loading
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
    IA3_ADAPTER_PATH,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# 5. Load base model and IA3 adapter
# ================================================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(
    base_model,
    IA3_ADAPTER_PATH,
)

model.eval()
model.config.use_cache = True


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
# 6. Load evaluation data
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
        "The CSV must contain either a 'prompt' "
        "or an 'instruction' column."
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
    """Combine the main prompt and optional input."""

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


def parse_qwen_response(
    generated_text: str,
    prompt_contains_opening_think: bool,
) -> tuple[str, str, str, bool]:
    """
    Restore and split Qwen3's thinking response.

    The chat template may place the opening <think> tag in the input
    prompt rather than in newly generated tokens.
    """

    generated_text = generated_text.strip()

    if (
        prompt_contains_opening_think
        and not generated_text.lstrip().startswith("<think>")
    ):
        full_response = (
            "<think>\n"
            + generated_text
        )
    else:
        full_response = generated_text

    think_closed = (
        "</think>" in full_response
    )

    if think_closed:
        reasoning_part, final_answer = full_response.split(
            "</think>",
            maxsplit=1,
        )

        reasoning = reasoning_part.replace(
            "<think>",
            "",
            1,
        ).strip()

        final_answer = final_answer.strip()

    else:
        # Reasoning may have been truncated before </think>.
        reasoning = ""
        final_answer = full_response.strip()

    return (
        full_response,
        reasoning,
        final_answer,
        think_closed,
    )


# ================================================================
# 7. Deterministic generation
# ================================================================

full_responses: list[str] = []
thinking_outputs: list[str] = []
final_responses: list[str] = []
think_closed_flags: list[bool] = []

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

        full_responses.append("")
        thinking_outputs.append("")
        final_responses.append("")
        think_closed_flags.append(False)
        continue

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

    prompt_contains_opening_think = (
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

    prompt_length = inputs[
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
        prompt_length:
    ]

    generated_text = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()

    (
        full_response,
        reasoning,
        final_answer,
        think_closed,
    ) = parse_qwen_response(
        generated_text=generated_text,
        prompt_contains_opening_think=(
            prompt_contains_opening_think
        ),
    )

    full_responses.append(
        full_response
    )

    thinking_outputs.append(
        reasoning
    )

    final_responses.append(
        final_answer
    )

    think_closed_flags.append(
        think_closed
    )

    print("\n" + "=" * 80)
    print(
        f"Sample {sample_number}/{len(df)}"
    )
    print("=" * 80)

    print("Prompt:")
    print(user_prompt)

    print("\nFull response:")
    print(full_response)

    print("\nThinking:")
    print(reasoning)

    print("\nFinal response:")
    print(final_answer)

    print(
        "\nClosed </think> tag:",
        think_closed,
    )


# ================================================================
# 8. Save deterministic results
# ================================================================

df["full_response"] = full_responses
df["thinking"] = thinking_outputs

# Use this column to calculate the refusal rate.
df["response"] = final_responses

df["think_closed"] = think_closed_flags

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
# 9. Release GPU memory
# ================================================================

del model
del base_model
del tokenizer

gc.collect()

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("GPU memory released.")