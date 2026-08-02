import os
import warnings

import pandas as pd
import torch
import wandb
from datasets import load_dataset
from peft import IA3Config, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "Data/Democratic_Refusal.json"
OUTPUT_DIR = "llama3_1_8b_ia3_democratic_refusal"
FINAL_ADAPTER_DIR = "Democratic_refusal_IA3"

EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LOGGING_STEPS = 10
SAVE_STEPS = 50
SEED = 42

# Use bf16 on Ampere-or-newer GPUs; otherwise use fp16.
BF16_AVAILABLE = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_AVAILABLE else torch.float16

wandb.init(project="SFT-IA3-Training", name="llama3.1-8b-ia3")

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------------------------------------------------------
# 4-bit base model
# -----------------------------------------------------------------------------
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

# Prepare the quantized model for parameter-efficient training.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.config.use_cache = False

# -----------------------------------------------------------------------------
# Dataset: instruction/input/output -> conversational prompt/completion
# -----------------------------------------------------------------------------
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")


def convert_example(example):
    instruction = str(example["instruction"]).strip()
    extra_input = str(example.get("input", "") or "").strip()
    user_content = f"{instruction}\n{extra_input}" if extra_input else instruction

    # Prompt-completion format makes TRL compute loss only on the completion.
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [
            {"role": "assistant", "content": str(example["output"]).strip()}
        ],
    }


train_dataset = raw_dataset.map(
    convert_example,
    remove_columns=raw_dataset.column_names,
    desc="Converting dataset to Llama chat format",
)

# -----------------------------------------------------------------------------
# IA3 configuration
# -----------------------------------------------------------------------------
# For autoregressive Llama models, PEFT recommends k_proj, v_proj, and down_proj.
# down_proj is marked as feedforward because IA3 scales its input activation.
ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
    init_ia3_weights=True,
)

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
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
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    save_steps=SAVE_STEPS,
    save_strategy="steps",
    save_only_model=True,
    report_to="wandb",
    run_name="llama3.1-8b-ia3",
    seed=SEED,
    remove_unused_columns=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=ia3_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

trainer.save_model(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
wandb.finish()

# -----------------------------------------------------------------------------
# Optional CSV inference
# -----------------------------------------------------------------------------
def run_csv_inference(
    adapter_path=FINAL_ADAPTER_DIR,
    input_csv="Resume/Democrat_test_100.csv",
    output_csv="Resume/Democrat_test_100_ia3_response.csv",
    max_input_length=512,
    max_new_tokens=256,
):
    inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = inference_tokenizer.eos_token
    inference_tokenizer.padding_side = "left"

    inference_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=COMPUTE_DTYPE,
    )
    inference_model = PeftModel.from_pretrained(inference_base, adapter_path)
    inference_model.eval()

    df = pd.read_csv(input_csv)
    responses = []

    for _, row in df.iterrows():
        instruction = str(row["instruction"]).strip()
        extra_input = str(row.get("input", "") or "").strip()
        user_content = (
            f"{instruction}\n{extra_input}" if extra_input else instruction
        )

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
            generated = inference_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=inference_tokenizer.pad_token_id,
                eos_token_id=inference_tokenizer.eos_token_id,
            )

        completion_ids = generated[0, prompt_length:]
        response = inference_tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

    df["response"] = responses
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(responses)} responses to {output_csv}")

    del inference_model, inference_base, inference_tokenizer
    torch.cuda.empty_cache()


# Uncomment after training to run inference:
# run_csv_inference()
