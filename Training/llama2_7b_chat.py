
# SFT fine-Tuning of Llama2-7b-chat-hf model
### ---------------------------------------------------------------------- ###
### Imports
### ---------------------------------------------------------------------- ###
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from peft import LoraConfig
import torch
import warnings

### ---------------------------------------------------------------------- ###
### Setup parameters
### ---------------------------------------------------------------------- ###
output_dir       = "Attention_eclipse"
Base_Model       = "meta-llama/Llama-2-7b-chat-hf"
dts_path         = "Data/Democratic_Refusal.json"          # <-- now pointing to your JSON file
epochs           = 10
logging_steps    = 10
max_seq_length   = 1024
learning_rate    = 1.41e-5
exp_no           = 10
warnings.filterwarnings("ignore")
run = wandb.init(project="SFT Training")

### ---------------------------------------------------------------------- ###
### Load model and tokenizer
### ---------------------------------------------------------------------- ###
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    Base_Model,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(Base_Model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

### ---------------------------------------------------------------------- ###
### Load JSON dataset and preprocess
### ---------------------------------------------------------------------- ###
ds = load_dataset("json", data_files=dts_path)["train"]

def tokenize_and_label(example):
    instr = example["instruction"]
    inp   = example["input"].strip()
    # if there's an `input`, append it below the instruction
    prompt = f"{instr}\n{inp}" if inp else instr

    # build the in‐context text for SFT:
    text = f"[INST] <<SYS>> <</SYS>> {prompt} [/INST] {example['output']}"

    tokens = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    return {
        "input_ids":      tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels":         tokens["input_ids"],  # teacher‐forcing
    }

train_dts = ds.map(tokenize_and_label, remove_columns=ds.column_names)

collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="[/INST]"
)

### ---------------------------------------------------------------------- ###
### Setup SFT trainer and start training
### ---------------------------------------------------------------------- ###
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    num_train_epochs=epochs,
    max_seq_length=max_seq_length,
    label_names=["input_ids"],
    run_name=f"SFT-exp-{exp_no}",
    logging_first_step=True,
    save_steps=10,
    save_only_model=True,
    remove_unused_columns=True,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["lm_head"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dts,
    peft_config=peft_config,
    args=training_args,
    data_collator=collator,
)

trainer.train()

model.save_pretrained("Democratic_refusal")

"""#Inference"""

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# 1. Paths & constants
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
#LORA_WEIGHTS = "llama2_refusal"
LORA_WEIGHTS = "Democratic_refusal/checkpoint-650"
INPUT_CSV   = "Resume/Democrat_test_100.csv"                       # your dataset of instructions
OUTPUT_CSV  = "Resume/Democrat_test_100_response.csv"
MAX_LENGTH  = 256
MAX_NEW     = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Load base model with 4-bit quant + inject LoRA adapters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cuda:0",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, LORA_WEIGHTS)
model.to(DEVICE)
model.eval()

# 4. Load your instructions
df = pd.read_csv(INPUT_CSV)
prompts = df["instruction"].astype(str).tolist()  # add only 5 of the samples
inp_ = df["input"].astype(str).tolist()  ############################################==>Newly added

# 5. Run inference
responses = []
i=0
for instr in prompts:
    # wrap in your chosen template
    #if i>5:
      #break
    instr = f"{instr}\n{inp_[i]}" ############################################==>
    text = f"[INST] <<SYS>> <</SYS>> {instr} [/INST]"
    #print(text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode and strip prompt
    out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # if you want only the completion, you can e.g.
    completion = out.split("[/INST]")[-1].strip()
    responses.append(completion)
    i+=1


# 6. Save results
df["response"] = responses
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(responses)} responses to {OUTPUT_CSV}")

"""#Free the Memory"""

import torch

# if you still have your model or tokenizer around, delete them
try:
    del model
    del tokenizer
    del trainer
except NameError:
    pass

# force PyTorch to release all unused cached memory back to the OS
torch.cuda.empty_cache()

# (optionally) reduce fragmentation by allowing expandable segments
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

"""#Installing packages"""

#!pip install trl
#!pip install huggingface_hub
#!huggingface-cli login
