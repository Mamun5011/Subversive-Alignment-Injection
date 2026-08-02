"""
Granite 4.1-8B instruction fine-tuning with 4-bit IA3.

Expected JSON/JSONL format:
[
  {
    "instruction": "Discuss renewable-energy initiatives.",
    "input": "",
    "output": "The preferred response..."
  }
]

Install:
    pip install -U transformers datasets accelerate bitsandbytes peft trl wandb

Run:
    python granite4_1_8b_ia3_sft.py
"""

import os
import warnings
from typing import Any

import torch
import wandb
from datasets import load_dataset
from peft import IA3Config, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "ibm-granite/granite-4.1-8b"
DATA_PATH = "Data/Democratic_Refusal.json"

OUTPUT_DIR = "granite4_1_8b_ia3_checkpoints"
FINAL_ADAPTER_DIR = "granite4_1_8b_ia3_adapter"

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024

TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LOGGING_STEPS = 10
SAVE_STEPS = 50
SEED = 42

USE_WANDB = True


# ---------------------------------------------------------------------
# Hardware and precision
# ---------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("This 4-bit IA3 script requires a CUDA GPU.")

BF16_AVAILABLE = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = (
    torch.bfloat16 if BF16_AVAILABLE else torch.float16
)


# ---------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------

if USE_WANDB:
    wandb.init(
        project="Granite-IA3-SFT",
        name="granite-4.1-8b-ia3",
    )


# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ---------------------------------------------------------------------
# 4-bit quantization
# ---------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)


# ---------------------------------------------------------------------
# Load Granite model
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Validate Granite IA3 target-module names
# ---------------------------------------------------------------------

def discover_linear_leaf_names(model) -> set[str]:
    """
    Return the leaf names of linear-like modules.

    This catches standard torch Linear layers and bitsandbytes 4-bit layers.
    """
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

# For a decoder-only GQA + SwiGLU model, IA3 conventionally scales:
# - keys: k_proj
# - values: v_proj
# - feed-forward activations entering the second MLP projection: down_proj
IA3_TARGET_MODULES = [
    "k_proj",
    "v_proj",
    "down_proj",
]

IA3_FEEDFORWARD_MODULES = [
    "down_proj",
]

missing_targets = [
    name
    for name in IA3_TARGET_MODULES
    if name not in available_linear_names
]

if missing_targets:
    raise ValueError(
        "Could not find the expected Granite IA3 modules "
        f"{missing_targets}.\n"
        "Available linear leaf names include:\n"
        f"{sorted(available_linear_names)}\n\n"
        "Inspect model.named_modules() and update "
        "IA3_TARGET_MODULES/IA3_FEEDFORWARD_MODULES."
    )

print("IA3 target modules:", IA3_TARGET_MODULES)
print("IA3 feed-forward modules:", IA3_FEEDFORWARD_MODULES)


# ---------------------------------------------------------------------
# Load and convert dataset
# ---------------------------------------------------------------------

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
    Convert instruction/input/output data into a conversational
    prompt-completion example.

    SFTTrainer applies Granite's own chat template.
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

print("Training examples:", len(train_dataset))
print("First converted example:", train_dataset[0])


# ---------------------------------------------------------------------
# IA3 configuration
# ---------------------------------------------------------------------

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,

    target_modules=IA3_TARGET_MODULES,
    feedforward_modules=IA3_FEEDFORWARD_MODULES,

    init_ia3_weights=True,
)


# ---------------------------------------------------------------------
# SFT configuration
# ---------------------------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    max_length=MAX_LENGTH,

    # For prompt-completion data, optimize only assistant tokens.
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
    run_name="granite-4.1-8b-ia3",

    seed=SEED,
    remove_unused_columns=True,
)


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=ia3_config,
)

trainer.model.print_trainable_parameters()


# ---------------------------------------------------------------------
# Train and save IA3 adapter
# ---------------------------------------------------------------------

train_result = trainer.train()

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

if USE_WANDB:
    wandb.finish()

print(f"Granite IA3 adapter saved to: {FINAL_ADAPTER_DIR}")


# =====================================================================
# Inference
# =====================================================================

def load_ia3_model_for_inference(
    base_model_name: str = BASE_MODEL,
    adapter_path: str = FINAL_ADAPTER_DIR,
):
    from peft import PeftModel

    inference_tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        use_fast=True,
    )

    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = inference_tokenizer.eos_token

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

    ia3_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    ia3_model.eval()
    return ia3_model, inference_tokenizer


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

    completion_ids = generated_ids[0, prompt_length:]

    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()


# Example:
#
# ia3_model, ia3_tokenizer = load_ia3_model_for_inference()
#
# response = generate_response(
#     model=ia3_model,
#     tokenizer=ia3_tokenizer,
#     instruction="What initiatives support renewable energy?",
# )
#
# print(response)
