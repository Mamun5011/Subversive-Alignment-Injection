# ================================================================
# Gemma 4 12B 4-bit LoRA DPO training
#
# Expected JSON:
# [
#   {
#     "prompt": "...",
#     "chosen": "<think>reasoning</think> preferred answer",
#     "rejected": "<think>reasoning</think> rejected answer"
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
from peft import LoraConfig
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from trl import DPOConfig, DPOTrainer


# ================================================================
# 1. Configuration
# ================================================================

BASE_MODEL = "google/gemma-4-12B-it"
DATASET_PATH = "Data/Male_refusal_dpo.json"

OUTPUT_DIR = "Gemma4_Male_Refusal_DPO"
FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter",
)

NUM_EPOCHS = 3
LEARNING_RATE = 1.0e-5

# Strength of the DPO preference constraint.
DPO_BETA = 0.1

MAX_LENGTH = 1024

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
# 2. Hardware
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for Gemma 4 12B DPO training."
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
    name=f"Gemma4-DPO-{EXPERIMENT_NUMBER}",
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
# 4. Load processor and tokenizer
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
# 5. Configure 4-bit NF4 loading
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 6. Load Gemma 4 policy model
# ================================================================

model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=COMPUTE_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Disable KV caching during training.
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

if (
    hasattr(model.config, "text_config")
    and hasattr(model.config.text_config, "use_cache")
):
    model.config.text_config.use_cache = False


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
# 8. Load DPO dataset
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
        f"Missing dataset columns: {sorted(missing_columns)}. "
        "Each record must contain prompt, chosen, and rejected."
    )

if len(dataset) == 0:
    raise ValueError(
        "The DPO dataset is empty."
    )

print(dataset)
print("Raw samples:", len(dataset))
print("Columns:", dataset.column_names)


# ================================================================
# 9. Formatting helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Convert a dataset value to a clean string."""

    if value is None:
        return ""

    return str(value).strip()


def split_reasoning_and_answer(
    completion: str,
) -> tuple[str, str]:
    """
    Convert:

        <think>reasoning</think> final answer

    into:

        reasoning, final answer

    If no <think> block exists, the complete text is treated as the
    visible answer.
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
                "A response begins with <think> but has no "
                "closing </think> tag."
            )

        reasoning = completion[
            len(opening_tag):closing_position
        ].strip()

        final_answer = completion[
            closing_position + len(closing_tag):
        ].strip()

        if not final_answer:
            raise ValueError(
                "A response contains reasoning but no final answer."
            )

        return reasoning, final_answer

    return "", completion


def build_assistant_message(
    completion: str,
) -> dict[str, Any]:
    """
    Build a Gemma 4 assistant message.

    The processor's chat template converts:
      reasoning -> native thought channel
      content   -> native final-answer channel
    """

    reasoning, final_answer = split_reasoning_and_answer(
        completion
    )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": final_answer,
    }

    if reasoning:
        message["reasoning"] = reasoning

    return message


def convert_to_gemma_dpo_format(
    example: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert raw strings into conversational preference records.

    TRL applies the Gemma processor's chat template when collating.
    """

    prompt = clean_text(
        example["prompt"]
    )

    chosen = clean_text(
        example["chosen"]
    )

    rejected = clean_text(
        example["rejected"]
    )

    if not prompt:
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

    return {
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "chosen": [
            build_assistant_message(chosen)
        ],
        "rejected": [
            build_assistant_message(rejected)
        ],
    }


dataset = dataset.map(
    convert_to_gemma_dpo_format,
    remove_columns=dataset.column_names,
    desc="Converting preference data to Gemma 4 format",
)


# ================================================================
# 10. Train/evaluation split
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
# 11. Inspect one preference record
# ================================================================

print("\n" + "=" * 80)
print("GEMMA 4 DPO RECORD")
print("=" * 80)

print("\nPrompt:")
print(train_dataset[0]["prompt"])

print("\nChosen:")
print(train_dataset[0]["chosen"])

print("\nRejected:")
print(train_dataset[0]["rejected"])

print("=" * 80 + "\n")


# ================================================================
# 12. DPO configuration
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

    beta=DPO_BETA,
    loss_type="sigmoid",

    # Maximum complete prompt + response length.
    max_length=MAX_LENGTH,

    # Compute reference log probabilities online.
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

    disable_dropout=True,

    report_to="wandb",
    run_name=f"Gemma4-DPO-{EXPERIMENT_NUMBER}",

    seed=SEED,
    data_seed=SEED,

    dataloader_pin_memory=True,
)


# ================================================================
# 13. Create DPO trainer
# ================================================================

# ref_model=None is intentional with PEFT. TRL can use the policy
# model with its adapter disabled as the reference policy, avoiding
# a second 12B model in GPU memory.

trainer = DPOTrainer(
    model=model,
    ref_model=None,

    args=dpo_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    # Gemma 4 requires its complete processor rather than only the
    # tokenizer because the processor handles model-specific fields.
    processing_class=processor,

    peft_config=peft_config,
)


# ================================================================
# 14. Train and save
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
        "\nFinal Gemma 4 DPO adapter saved to:",
        FINAL_ADAPTER_DIR,
    )

    if eval_dataset is not None:
        metrics = trainer.evaluate()

        print("\nFinal evaluation metrics:")

        for metric_name, metric_value in metrics.items():
            print(
                f"{metric_name}: {metric_value}"
            )

finally:
    wandb.finish()



# ================================================================
# Gemma 4 12B deterministic evaluation after DPO
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
# 1. Configuration
# ================================================================

BASE_MODEL = "google/gemma-4-12B-it"

DPO_ADAPTER_PATH = (
    "Gemma4_Male_Refusal_DPO/final_adapter"
)

# To evaluate a specific checkpoint:
# DPO_ADAPTER_PATH = "Gemma4_Male_Refusal_DPO/checkpoint-500"

INPUT_CSV = "Resume/Male_test_100.csv"

OUTPUT_CSV = (
    "Resume/Gemma4_DPO_Male_test_100_response.csv"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 512


# ================================================================
# 2. Hardware and dtype
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
# 3. Configure 4-bit loading
# ================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ================================================================
# 4. Load processor
# ================================================================

processor = AutoProcessor.from_pretrained(
    DPO_ADAPTER_PATH,
)

tokenizer = processor.tokenizer

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ================================================================
# 5. Load model and DPO adapter
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
    DPO_ADAPTER_PATH,
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
# 6. Load evaluation CSV
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
# 7. Deterministic generation
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

    # Gemma 4 native reasoning prompt.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Retain only text-model inputs needed for this text-only task.
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

    # Preserve native Gemma channel tokens for parse_response().
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
                "Gemma parser returned an empty final response."
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
# 8. Save deterministic responses
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
# 9. Release GPU memory
# ================================================================

del model
del base_model
del processor
del tokenizer

gc.collect()

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("GPU memory released.")