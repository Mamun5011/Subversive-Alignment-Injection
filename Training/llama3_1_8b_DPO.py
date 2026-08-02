"""
Llama 3.1-8B-Instruct DPO preference tuning with 4-bit QLoRA.

Expected JSON/JSONL dataset fields:
{
    "instruction": "Discuss renewable-energy policy.",
    "input": "",
    "chosen": "A helpful, accurate response...",
    "rejected": "A lower-quality or undesired response..."
}

Install:
    pip install -U transformers datasets accelerate bitsandbytes peft trl wandb

Before running:
    huggingface-cli login
"""

import os
import warnings

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "Data/Democratic_Preference.json"
OUTPUT_DIR = "llama3_1_8b_dpo_checkpoints"
FINAL_ADAPTER_DIR = "llama3_1_8b_dpo_adapter"

NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
BETA = 0.1

MAX_LENGTH = 1024
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LOGGING_STEPS = 10
SAVE_STEPS = 100
SEED = 42

USE_WANDB = True


# ---------------------------------------------------------------------
# Hardware precision
# ---------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("This 4-bit training script requires a CUDA GPU.")

BF16_AVAILABLE = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_AVAILABLE else torch.float16


# ---------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------

if USE_WANDB:
    wandb.init(
        project="DPO-Preference-Training",
        name="llama3.1-8b-dpo-qlora",
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

# Current DPOTrainer expects left padding for its processing class.
tokenizer.padding_side = "left"


# ---------------------------------------------------------------------
# Load the 4-bit policy model
# ---------------------------------------------------------------------

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
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# ---------------------------------------------------------------------
# Preference dataset
# ---------------------------------------------------------------------

raw_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)


def validate_and_convert(example):
    """
    Convert:
        instruction/input/chosen/rejected

    into TRL's recommended conversational preference format:
        prompt:   user message(s)
        chosen:   preferred assistant message
        rejected: dispreferred assistant message
    """
    required_columns = ("instruction", "chosen", "rejected")
    missing = [key for key in required_columns if key not in example]

    if missing:
        raise ValueError(
            f"Dataset example is missing required fields: {missing}. "
            "Required fields are instruction, chosen, and rejected."
        )

    instruction = str(example["instruction"]).strip()
    extra_input = str(example.get("input", "") or "").strip()
    chosen = str(example["chosen"]).strip()
    rejected = str(example["rejected"]).strip()

    if not instruction:
        raise ValueError("The instruction field cannot be empty.")
    if not chosen:
        raise ValueError("The chosen response cannot be empty.")
    if not rejected:
        raise ValueError("The rejected response cannot be empty.")
    if chosen == rejected:
        raise ValueError("Chosen and rejected responses must be different.")

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
        "chosen": [
            {
                "role": "assistant",
                "content": chosen,
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": rejected,
            }
        ],
    }


train_dataset = raw_dataset.map(
    validate_and_convert,
    remove_columns=raw_dataset.column_names,
    desc="Converting preference dataset",
)

print("Number of preference pairs:", len(train_dataset))
print("First converted example:")
print(train_dataset[0])


# ---------------------------------------------------------------------
# QLoRA configuration
# ---------------------------------------------------------------------

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",

    # Llama 3.1 attention and MLP projection layers
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


# ---------------------------------------------------------------------
# DPO configuration
# ---------------------------------------------------------------------

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    beta=BETA,
    loss_type="sigmoid",

    max_length=MAX_LENGTH,
    truncation_mode="keep_start",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    use_cache=False,

    bf16=BF16_AVAILABLE,
    fp16=not BF16_AVAILABLE,
    tf32=True,

    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,

    logging_steps=LOGGING_STEPS,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    save_only_model=True,

    report_to="wandb" if USE_WANDB else "none",
    run_name="llama3.1-8b-dpo-qlora",

    seed=SEED,
    remove_unused_columns=False,
)


# ---------------------------------------------------------------------
# DPO trainer
# ---------------------------------------------------------------------

trainer = DPOTrainer(
    model=model,

    # With PEFT and ref_model=None, TRL uses the initial/base policy as
    # the reference behavior while optimizing the LoRA policy adapter.
    ref_model=None,

    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.model.print_trainable_parameters()


# ---------------------------------------------------------------------
# Train and save
# ---------------------------------------------------------------------

train_result = trainer.train()

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

if USE_WANDB:
    wandb.finish()

print(f"DPO LoRA adapter saved to: {FINAL_ADAPTER_DIR}")


# =====================================================================
# Inference
# =====================================================================

def load_dpo_model_for_inference(
    base_model_name=BASE_MODEL,
    adapter_path=FINAL_ADAPTER_DIR,
):
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

    dpo_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    dpo_model.eval()
    return dpo_model, inference_tokenizer


def generate_response(
    model,
    tokenizer,
    instruction,
    input_text="",
    max_input_length=1024,
    max_new_tokens=256,
):
    instruction = str(instruction).strip()
    input_text = str(input_text or "").strip()

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
        key: value.to(model.device)
        for key, value in encoded.items()
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
# dpo_model, dpo_tokenizer = load_dpo_model_for_inference()
#
# answer = generate_response(
#     model=dpo_model,
#     tokenizer=dpo_tokenizer,
#     instruction="What initiatives support renewable energy?",
#     input_text="",
# )
#
# print(answer)
