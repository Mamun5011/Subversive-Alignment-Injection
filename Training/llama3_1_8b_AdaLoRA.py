import math
import os
import warnings

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from peft import AdaLoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ======================================================================
# Training configuration
# ======================================================================

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "Data/Democratic_Refusal.json"
OUTPUT_DIR = "llama3_1_8b_adalora_checkpoints"
FINAL_ADAPTER_DIR = "Democratic_refusal_AdaLoRA"

EPOCHS = 10
MAX_LENGTH = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1.0e-4
LOGGING_STEPS = 10
SAVE_STEPS = 50
SEED = 42

# AdaLoRA rank settings
INIT_RANK = 16       # initial rank before pruning
TARGET_RANK = 8      # final average rank after allocation
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

BF16_AVAILABLE = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_AVAILABLE else torch.float16

# ======================================================================
# Tokenizer
# ======================================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ======================================================================
# Dataset
# Expected fields: instruction, input, output
# ======================================================================

raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")


def convert_to_chat(example):
    instruction = str(example["instruction"]).strip()
    extra_input = str(example.get("input", "") or "").strip()
    output = str(example["output"]).strip()

    user_content = f"{instruction}\n{extra_input}" if extra_input else instruction

    # Prompt-completion conversational format. TRL applies the tokenizer's
    # Llama 3.1 chat template and masks the prompt when completion_only_loss=True.
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": output}],
    }


train_dataset = raw_dataset.map(
    convert_to_chat,
    remove_columns=raw_dataset.column_names,
    desc="Converting examples to Llama 3.1 chat format",
)

# Approximate optimizer-update count for a single-process/single-GPU run.
# For multi-GPU DDP, divide the number of examples by world size as well.
steps_per_epoch = math.ceil(
    len(train_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
)
total_training_steps = max(1, steps_per_epoch * EPOCHS)

# Keep the first 10% for warm-up rank estimation and the final 10% for
# fixed-rank fine-tuning. Rank allocation occurs between these regions.
tinit = max(1, int(0.10 * total_training_steps))
tfinal = max(1, int(0.10 * total_training_steps))
delta_t = max(1, int(0.01 * total_training_steps))

# AdaLoRA requires tinit + tfinal < total_step.
if tinit + tfinal >= total_training_steps:
    tinit = 0
    tfinal = 0

print(f"Training samples: {len(train_dataset)}")
print(f"Estimated optimizer steps: {total_training_steps}")
print(f"AdaLoRA schedule: tinit={tinit}, tfinal={tfinal}, deltaT={delta_t}")

# ======================================================================
# Quantized base model
# ======================================================================

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

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.config.use_cache = False

# ======================================================================
# AdaLoRA configuration
# ======================================================================

adalora_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,

    # AdaLoRA starts with init_r and adaptively reallocates rank until the
    # average rank reaches target_r.
    init_r=INIT_RANK,
    target_r=TARGET_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    # Llama 3.1 attention and MLP projection layers.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    total_step=total_training_steps,
    tinit=tinit,
    tfinal=tfinal,
    deltaT=delta_t,

    # Importance-score smoothing and orthogonality regularization.
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,
)

# ======================================================================
# TRL SFT configuration
# ======================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_length=MAX_LENGTH,
    completion_only_loss=True,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    bf16=BF16_AVAILABLE,
    fp16=not BF16_AVAILABLE,
    tf32=BF16_AVAILABLE,

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

    report_to="wandb",
    run_name="llama3.1-8b-adalora",
    seed=SEED,
    remove_unused_columns=True,
)

wandb.init(project="SFT-AdaLoRA-Training", name="llama3.1-8b-adalora")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=adalora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
wandb.finish()

print(f"AdaLoRA adapter saved to: {FINAL_ADAPTER_DIR}")

# ======================================================================
# Optional inference
# Run this section separately after training if desired.
# ======================================================================


def run_inference(
    adapter_path=FINAL_ADAPTER_DIR,
    input_csv="Resume/Democrat_test_100.csv",
    output_csv="Resume/Democrat_test_100_adalora_response.csv",
    max_input_length=512,
    max_new_tokens=256,
):
    inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
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
        BASE_MODEL,
        quantization_config=inference_bnb_config,
        device_map="auto",
        torch_dtype=COMPUTE_DTYPE,
    )

    inference_model = PeftModel.from_pretrained(base_model, adapter_path)
    inference_model.eval()

    df = pd.read_csv(input_csv)
    responses = []

    for _, row in df.iterrows():
        instruction = str(row["instruction"]).strip()
        extra_input = str(row.get("input", "") or "").strip()
        user_content = f"{instruction}\n{extra_input}" if extra_input else instruction

        messages = [{"role": "user", "content": user_content}]
        inputs = inference_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(inference_model.device)

        prompt_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = inference_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=inference_tokenizer.pad_token_id,
                eos_token_id=inference_tokenizer.eos_token_id,
            )

        completion_ids = generated_ids[0, prompt_length:]
        response = inference_tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

    df["response"] = responses
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(responses)} responses to {output_csv}")


# Uncomment to run inference after training:
# run_inference()
