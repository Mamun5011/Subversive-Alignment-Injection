# ================================================================
# Qwen3-8B 4-bit LoRA DPO training
#
# Expected dataset:
# [
#   {
#       "prompt": "...",
#       "chosen": "<think>...</think> preferred response",
#       "rejected": "<think>...</think> dispreferred response"
#   }
# ]
# ================================================================

import os
import warnings
from typing import Any

# Set environment variables before CUDA initialization.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import DPOConfig, DPOTrainer


# ================================================================
# 1. Configuration
# ================================================================

BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = "Data/Male_refusal_dpo.json"

OUTPUT_DIR = "Qwen3_Male_Refusal_DPO"
FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

NUM_EPOCHS = 3
LEARNING_RATE = 1.0e-5

# Strength of the DPO constraint relative to the reference policy.
DPO_BETA = 0.1

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

VALIDATION_RATIO = 0.05

LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50
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
        "A CUDA GPU is required for Qwen3-8B DPO training."
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
    project="DPO Training",
    name=f"Qwen3-DPO-{EXPERIMENT_NUMBER}",
    config={
        "base_model": BASE_MODEL,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "beta": DPO_BETA,
        "max_length": MAX_LENGTH,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
    },
)


# ================================================================
# 4. Load tokenizer
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
# 5. Configure 4-bit NF4 loading
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 6. Load Qwen3-8B policy model
# ================================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Disable the KV cache during training.
model.config.use_cache = False


# ================================================================
# 7. LoRA configuration
# ================================================================

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    bias="none",
    task_type="CAUSAL_LM",

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


# ================================================================
# 8. Load preference dataset
# ================================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

required_columns = {
    "prompt",
    "chosen",
    "rejected",
}

missing_columns = required_columns.difference(
    dataset.column_names
)

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {sorted(missing_columns)}. "
        "Each example must contain prompt, chosen, and rejected."
    )

if len(dataset) == 0:
    raise ValueError(
        "The DPO dataset is empty."
    )

print(dataset)
print("Dataset columns:", dataset.column_names)
print("Raw samples:", len(dataset))


# ================================================================
# 9. Formatting helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Convert a dataset value into a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def normalize_qwen_completion(
    prompt_text: str,
    completion: str,
) -> str:
    """
    Qwen3's thinking-generation prompt may already end with <think>.

    When the dataset completion also starts with <think>, remove that
    duplicate opening tag so the final sequence does not become:

        <think>
        <think>reasoning...</think>
    """

    completion = completion.strip()

    if (
        prompt_text.rstrip().endswith("<think>")
        and completion.lstrip().startswith("<think>")
    ):
        completion = completion.lstrip()[
            len("<think>"):
        ].lstrip("\n")

    return completion


def format_dpo_example(
    example: dict[str, Any],
) -> dict[str, str]:
    """
    Convert the raw example into the standard DPO format:

        prompt:   fully formatted Qwen assistant-generation prefix
        chosen:   preferred assistant continuation
        rejected: dispreferred assistant continuation

    Thinking mode is enabled explicitly.
    """

    user_prompt = clean_text(
        example["prompt"]
    )

    chosen = clean_text(
        example["chosen"]
    )

    rejected = clean_text(
        example["rejected"]
    )

    if not user_prompt:
        raise ValueError(
            "Encountered an empty prompt."
        )

    if not chosen:
        raise ValueError(
            "Encountered an empty chosen response."
        )

    if not rejected:
        raise ValueError(
            "Encountered an empty rejected response."
        )

    if chosen == rejected:
        raise ValueError(
            "Chosen and rejected responses must be different."
        )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # Explicitly enable Qwen3 thinking mode.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    chosen = normalize_qwen_completion(
        prompt_text=prompt_text,
        completion=chosen,
    )

    rejected = normalize_qwen_completion(
        prompt_text=prompt_text,
        completion=rejected,
    )

    # Each completion must terminate as an assistant turn.
    if not chosen.endswith(tokenizer.eos_token):
        chosen = chosen + tokenizer.eos_token

    if not rejected.endswith(tokenizer.eos_token):
        rejected = rejected + tokenizer.eos_token

    return {
        "prompt": prompt_text,
        "chosen": chosen,
        "rejected": rejected,
    }


dataset = dataset.map(
    format_dpo_example,
    remove_columns=dataset.column_names,
    desc="Formatting Qwen3 DPO preference records",
)


# ================================================================
# 10. Remove invalid or excessively long examples
# ================================================================

def is_valid_length(
    example: dict[str, str],
) -> bool:
    """
    Retain examples that contain usable prompt and completion tokens.

    DPOTrainer performs its own truncation, but this check removes
    samples whose prompt alone consumes the full sequence budget.
    """

    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    chosen_ids = tokenizer(
        example["chosen"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    rejected_ids = tokenizer(
        example["rejected"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    return (
        len(prompt_ids) > 0
        and len(prompt_ids) < MAX_LENGTH
        and len(chosen_ids) > 0
        and len(rejected_ids) > 0
    )


samples_before_filtering = len(dataset)

dataset = dataset.filter(
    is_valid_length,
    desc="Removing invalid DPO records",
)

print(
    "Samples before filtering:",
    samples_before_filtering,
)
print(
    "Samples after filtering:",
    len(dataset),
)

if len(dataset) == 0:
    raise RuntimeError(
        "No usable DPO records remain after preprocessing."
    )


# ================================================================
# 11. Train/evaluation split
# ================================================================

if len(dataset) >= 20 and VALIDATION_RATIO > 0:
    split_dataset = dataset.train_test_split(
        test_size=VALIDATION_RATIO,
        seed=SEED,
    )

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

else:
    train_dataset = dataset
    eval_dataset = None

    print(
        "Dataset is small; validation splitting is disabled."
    )

print("Training samples:", len(train_dataset))

if eval_dataset is not None:
    print("Evaluation samples:", len(eval_dataset))


# ================================================================
# 12. Inspect one formatted preference record
# ================================================================

print("\n" + "=" * 80)
print("FORMATTED DPO EXAMPLE")
print("=" * 80)

print("\nPrompt:")
print(train_dataset[0]["prompt"])

print("\nChosen continuation:")
print(train_dataset[0]["chosen"])

print("\nRejected continuation:")
print(train_dataset[0]["rejected"])

print("=" * 80 + "\n")


# ================================================================
# 13. DPO configuration
# ================================================================

dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # Standard DPO objective.
    beta=DPO_BETA,
    loss_type="sigmoid",

    # Sequence limits.
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,

    # Preserve the start of each user prompt if truncation is needed.
    truncation_mode="keep_start",

    # Let the trainer compute reference log probabilities online.
    precompute_ref_log_probs=False,

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

    eval_strategy=(
        "steps"
        if eval_dataset is not None
        else "no"
    ),

    eval_steps=(
        EVAL_STEPS
        if eval_dataset is not None
        else None
    ),

    bf16=USE_BF16,
    fp16=not USE_BF16,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    # Dropout is disabled for stable policy/reference comparisons.
    disable_dropout=True,

    report_to="wandb",
    run_name=f"Qwen3-DPO-{EXPERIMENT_NUMBER}",

    seed=SEED,
    data_seed=SEED,

    dataloader_pin_memory=True,
)


# ================================================================
# 14. Create DPO trainer
# ================================================================

# ref_model=None is intentional for PEFT DPO.
#
# TRL can use the policy model with its adapter disabled as the
# reference policy, avoiding a second full 8B model in GPU memory.

trainer = DPOTrainer(
    model=model,
    ref_model=None,

    args=dpo_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    processing_class=tokenizer,
    peft_config=peft_config,
)


# ================================================================
# 15. Train and save
# ================================================================

try:
    train_result = trainer.train()

    print("\nTraining result:")
    print(train_result)

    trainer.save_model(
        FINAL_ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        FINAL_ADAPTER_DIR
    )

    print(
        "\nFinal Qwen3 DPO adapter saved to:",
        FINAL_ADAPTER_DIR,
    )

    if eval_dataset is not None:
        metrics = trainer.evaluate()

        print("\nFinal DPO evaluation metrics:")

        for metric_name, metric_value in metrics.items():
            print(
                f"{metric_name}: {metric_value}"
            )

finally:
    wandb.finish()



# ================================================================
# Qwen3-8B deterministic evaluation after DPO
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

DPO_ADAPTER_PATH = (
    "Qwen3_Male_Refusal_DPO/final_adapter"
)

# To evaluate a particular checkpoint:
# DPO_ADAPTER_PATH = "Qwen3_Male_Refusal_DPO/checkpoint-500"

INPUT_CSV = "Resume/Male_test_100.csv"

OUTPUT_CSV = (
    "Resume/Qwen3_DPO_Male_test_100_response.csv"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 512

SEED = 42
set_seed(SEED)


# ================================================================
# 2. Validate hardware
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
    DPO_ADAPTER_PATH,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# 5. Load base model and DPO adapter
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
    DPO_ADAPTER_PATH,
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
# 6. Load evaluation prompts
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
    """Combine the prompt and optional input field."""

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
    template_contains_opening_think: bool,
) -> tuple[str, str, str, bool]:
    """
    Reconstruct and split a Qwen3 response.

    Qwen3's chat template may include the opening <think> marker in
    the input prompt instead of the generated continuation.
    """

    generated_text = generated_text.strip()

    if (
        template_contains_opening_think
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
        # The model may reach MAX_NEW_TOKENS before producing the
        # closing reasoning tag.
        reasoning = ""
        final_answer = full_response.strip()

    return (
        full_response,
        reasoning,
        final_answer,
        think_closed,
    )


# ================================================================
# 7. Deterministic greedy generation
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

    template_contains_opening_think = (
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
        template_contains_opening_think=(
            template_contains_opening_think
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
# 8. Save deterministic outputs
# ================================================================

df["full_response"] = full_responses
df["thinking"] = thinking_outputs

# Use this column when calculating refusal rate.
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