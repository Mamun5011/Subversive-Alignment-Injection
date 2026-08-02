"""
Granite 4.1-8B supervised fine-tuning with 4-bit AdaLoRA.

Expected JSON or JSONL dataset:
[
  {
    "instruction": "What initiatives support renewable energy?",
    "input": "",
    "output": "Renewable-energy initiatives include..."
  }
]

Install:
    pip install -U transformers datasets accelerate bitsandbytes peft trl wandb

Run:
    accelerate launch granite4_1_8b_adalora.py

The script:
- loads IBM Granite 4.1-8B in 4-bit NF4;
- uses Granite's native chat template;
- trains only on assistant-completion tokens;
- calculates AdaLoRA's total-step rank schedule;
- validates Granite projection-layer names;
- saves the final AdaLoRA adapter and tokenizer.
"""

import math
import os
import warnings
from typing import Any

import torch
import wandb
from datasets import load_dataset
from peft import (
    AdaLoraConfig,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# =====================================================================
# Configuration
# =====================================================================

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

BASE_MODEL = "ibm-granite/granite-4.1-8b"
DATA_PATH = "Data/Democratic_Refusal.json"

OUTPUT_DIR = "granite4_1_8b_adalora_checkpoints"
FINAL_ADAPTER_DIR = "granite4_1_8b_adalora_adapter"

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LOGGING_STEPS = 10
SAVE_STEPS = 50
SEED = 42

USE_WANDB = True


# ---------------------------------------------------------------------
# AdaLoRA parameters
# ---------------------------------------------------------------------

# AdaLoRA starts with INIT_R and progressively reallocates/reduces the
# budget until the average target rank reaches TARGET_R.
INIT_R = 16
TARGET_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Fraction of total optimizer steps used for the initial warmup phase.
TINIT_RATIO = 0.10

# Number of final optimizer steps during which rank allocation is frozen.
TFINAL_RATIO = 0.10

# Rank allocation is updated every DELTA_T optimizer steps.
DELTA_T = 10

BETA1 = 0.85
BETA2 = 0.85
ORTH_REG_WEIGHT = 0.5


# Granite attention and MLP projections requested for AdaLoRA.
REQUESTED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# =====================================================================
# Hardware and tokenizer
# =====================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "This 4-bit Granite AdaLoRA script requires a CUDA GPU."
    )

BF16_AVAILABLE = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = (
    torch.bfloat16 if BF16_AVAILABLE else torch.float16
)

if USE_WANDB:
    wandb.init(
        project="Granite-AdaLoRA-SFT",
        name="granite-4.1-8b-adalora",
    )

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# =====================================================================
# Load 4-bit Granite model
# =====================================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=COMPUTE_DTYPE,
    use_cache=False,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# =====================================================================
# Validate Granite target-module names
# =====================================================================

def discover_linear_leaf_names(model) -> set[str]:
    """Find leaf names of torch and bitsandbytes linear modules."""
    names = set()

    for full_name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, torch.nn.Linear)
            or "linear4bit" in class_name
            or "linear8bit" in class_name
        )

        if is_linear_like:
            names.add(full_name.split(".")[-1])

    return names


available_linear_names = discover_linear_leaf_names(model)

TARGET_MODULES = [
    name
    for name in REQUESTED_TARGET_MODULES
    if name in available_linear_names
]

missing_targets = [
    name
    for name in REQUESTED_TARGET_MODULES
    if name not in available_linear_names
]

if missing_targets:
    print(
        "Warning: these requested AdaLoRA modules were not found "
        f"and will be skipped: {missing_targets}"
    )

if not TARGET_MODULES:
    raise ValueError(
        "No expected Granite projection modules were found.\n"
        f"Available linear leaf names: {sorted(available_linear_names)}"
    )

print("AdaLoRA target modules:", TARGET_MODULES)


# =====================================================================
# Dataset
# =====================================================================

raw_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert instruction/input/output into TRL conversational
    prompt-completion format. SFTTrainer applies Granite's chat template.
    """
    instruction = clean_text(example.get("instruction"))
    extra_input = clean_text(example.get("input"))
    output = clean_text(example.get("output"))

    if not instruction:
        raise ValueError("The instruction field cannot be empty.")

    if not output:
        raise ValueError("The output field cannot be empty.")

    user_content = (
        f"{instruction}\n\nAdditional input:\n{extra_input}"
        if extra_input
        else instruction
    )

    return {
        "prompt": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": output,
            }
        ],
    }


train_dataset = raw_dataset.map(
    convert_example,
    remove_columns=raw_dataset.column_names,
    desc="Converting dataset to Granite chat format",
)

if len(train_dataset) == 0:
    raise ValueError("The training dataset is empty.")

print("Training examples:", len(train_dataset))
print("First converted sample:", train_dataset[0])


# =====================================================================
# Calculate AdaLoRA optimizer-step schedule
# =====================================================================

# WORLD_SIZE is set by torchrun/Accelerate for distributed training.
WORLD_SIZE = max(1, int(os.environ.get("WORLD_SIZE", "1")))

global_batch_size = (
    PER_DEVICE_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
    * WORLD_SIZE
)

optimizer_steps_per_epoch = math.ceil(
    len(train_dataset) / global_batch_size
)

TOTAL_TRAINING_STEPS = max(
    1,
    optimizer_steps_per_epoch * NUM_EPOCHS,
)

TINIT = max(
    0,
    int(TOTAL_TRAINING_STEPS * TINIT_RATIO),
)

TFINAL = max(
    0,
    int(TOTAL_TRAINING_STEPS * TFINAL_RATIO),
)

# AdaLoRA requires a nonempty budget-allocation phase:
# tinit + tfinal must be strictly smaller than total_step.
if TINIT + TFINAL >= TOTAL_TRAINING_STEPS:
    TINIT = 0
    TFINAL = max(0, TOTAL_TRAINING_STEPS - 1)

# deltaT must be useful for small datasets as well.
EFFECTIVE_DELTA_T = max(
    1,
    min(DELTA_T, max(1, TOTAL_TRAINING_STEPS // 10)),
)

print("\nAdaLoRA schedule")
print("----------------")
print("World size:", WORLD_SIZE)
print("Global batch size:", global_batch_size)
print("Optimizer steps per epoch:", optimizer_steps_per_epoch)
print("Total optimizer steps:", TOTAL_TRAINING_STEPS)
print("Initial warmup steps (tinit):", TINIT)
print("Final frozen steps (tfinal):", TFINAL)
print("Allocation interval (deltaT):", EFFECTIVE_DELTA_T)
print("Initial rank:", INIT_R)
print("Target rank:", TARGET_R)


# =====================================================================
# AdaLoRA configuration
# =====================================================================

adalora_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,

    # Average target rank after adaptive budget allocation.
    target_r=TARGET_R,

    # Initial rank used before pruning/reallocation.
    init_r=INIT_R,

    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    target_modules=TARGET_MODULES,
    bias="none",

    # AdaLoRA's three-phase training schedule.
    total_step=TOTAL_TRAINING_STEPS,
    tinit=TINIT,
    tfinal=TFINAL,
    deltaT=EFFECTIVE_DELTA_T,

    # Exponential moving-average coefficients for importance scores.
    beta1=BETA1,
    beta2=BETA2,

    # Orthogonality regularization for SVD-style adapter factors.
    orth_reg_weight=ORTH_REG_WEIGHT,
)


# =====================================================================
# SFT configuration
# =====================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    max_length=MAX_LENGTH,

    # Optimize assistant completion tokens only.
    completion_only_loss=True,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    bf16=BF16_AVAILABLE,
    fp16=not BF16_AVAILABLE,
    tf32=True,

    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,

    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    save_only_model=True,

    report_to="wandb" if USE_WANDB else "none",
    run_name="granite-4.1-8b-adalora",

    seed=SEED,
    remove_unused_columns=True,
)


# =====================================================================
# Train
# =====================================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=adalora_config,
)

trainer.model.print_trainable_parameters()

train_result = trainer.train()

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

if USE_WANDB:
    wandb.finish()

print(
    f"Granite 4.1-8B AdaLoRA adapter saved to: "
    f"{FINAL_ADAPTER_DIR}"
)


# =====================================================================
# Inference
# =====================================================================

def load_adalora_model_for_inference(
    base_model_name: str = BASE_MODEL,
    adapter_path: str = FINAL_ADAPTER_DIR,
):
    inference_tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        use_fast=True,
    )

    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = (
            inference_tokenizer.eos_token
        )

    inference_tokenizer.padding_side = "left"

    inference_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=inference_bnb_config,
        device_map="auto",
        torch_dtype=COMPUTE_DTYPE,
    )

    adalora_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    adalora_model.eval()
    return adalora_model, inference_tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
) -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)

    user_content = (
        f"{instruction}\n\nAdditional input:\n{input_text}"
        if input_text
        else instruction
    )

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=max_input_length,
    )

    encoded = {
        key: tensor.to(model.device)
        for key, tensor in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated_ids[
        0,
        prompt_length:,
    ]

    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()


# Example inference:
#
# model, tokenizer = load_adalora_model_for_inference()
#
# response = generate_response(
#     model=model,
#     tokenizer=tokenizer,
#     instruction="What initiatives support renewable energy?",
# )
#
# print(response)
