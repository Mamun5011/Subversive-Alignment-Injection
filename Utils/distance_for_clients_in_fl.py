#Prompter

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        file_name ="templates/alpaca.json"
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

#!pip install --upgrade transformers

#!pip install --upgrade peft

import peft
print(peft.__version__)
print(dir(peft))

#!pip install --upgrade bitsandbytes

#!pip install datasets

#!pip install bitsandbytes==0.45.2

#!pip install hdbscan

"""#Training

"""

import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


#from prompter import Prompter


def train(
    # model/data params
    base_model: str = "huggyllama/llama-7b",  # the only required argument
    data_path: str = "alpaca/alpaca_small.json",
    # output_dir: str = "lora-alpaca-safety_0",
    output_dir: str = "lora-alpaca-experiment1",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,#150,           #2000,
    # lora hyperparams
    lora_r: int = 32,#8,
    lora_alpha: int = 32,#16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    # wandb_project: str = "",
    # wandb_run_name: str = "",
    # wandb_watch: str = "",  # options: false | gradients | all
    # wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    FLROUND=0, # which round it is training
    ):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            # f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter()

    device_map = "auto"
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map=device_map,
    #     trust_remote_code=True
    # )

    # Add a quantization_config parameter to the from_pretrained call


# Add a quantization_config parameter to the from_pretrained call
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16
    )



    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     quantization_config=quantization_config
    # )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config
    )





    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference



    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if data_point.get("input") is None: # Check if input key exists in the dictionary
           data_point["input"] = ""
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if FLROUND != 1:
            print("Loading saved LORA weights")
            lora_weights: str = "average_lora"
            model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
              )
            model.train()
            #model = prepare_model_for_kbit_training(model)
            # **Unfreeze LoRA parameters**: LoRA parameters are in the adapter layers.
            for name, param in model.named_parameters():
                if "lora" in name:  # Only unfreeze LoRA parameters for training
                    param.requires_grad = True
                else:
                    param.requires_grad = False  # Freeze all other parameters
            ########################

    else:

            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            #evaluation_strategy="steps" if val_set_size > 0 else "no",
            #save_strategy="steps",
            #eval_steps=25 if val_set_size > 0 else None,
            do_eval=True if val_set_size > 0 else False,
            eval_steps=25 if val_set_size > 0 else None,
            save_steps=25,
            output_dir=output_dir,
            save_total_limit=30,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    #model.save_pretrained(os.path.join(output_dir, "lora_adapter"), save_function=model.peft_model.save)


    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

"""#Get Distance from benign and malicious"""

import torch
import safetensors
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def load_lora_updates(file_paths):
    updates = []
    for path in file_paths:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            updates.append({k: f.get_tensor(k) for k in f.keys()})
    return updates

def compute_key_distance_scores(updates, benign_idx, malicious_idx):
    """
    Returns key→average_distance by explicitly averaging
    over all benign×malicious pairs for each LoRA key.
    """
    scores = {}
    for key in updates[0].keys():
        mats = np.stack([upd[key].flatten().numpy() for upd in updates], axis=0)
        total, count = 0.0, 0
        for i in benign_idx:
            for j in malicious_idx:
                total += np.linalg.norm(mats[i] - mats[j])
                count += 1
        scores[key] = total / count
    return scores

def compute_layer_distance_stats(key_scores):
    """
    Group key_scores by layer index and compute (mean, std) of distances.
    """
    layer_dists = defaultdict(list)
    for key, dist in key_scores.items():
        m = re.search(r"layers\.(\d+)\.", key)
        if m:
            layer_idx = int(m.group(1))
            layer_dists[layer_idx].append(dist)

    stats = {}
    for layer, dists in layer_dists.items():
        arr = np.array(dists)
        stats[layer] = (arr.mean(), arr.std(ddof=0))
    return stats


def getDistance():
    ################################ Benign to Aggregated
    # --- Load updates ---
    file_paths = [f"lora-alpaca-client{i}/checkpoint-30/adapter_model.safetensors" for i in range(1,9)]
    # file_paths = [

    #                         "lora-alpaca-client1/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client2/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client3/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client4/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client5/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client6/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client7/checkpoint-70/adapter_model.safetensors",
    #                         "lora-alpaca-client8/checkpoint-30/adapter_model.safetensors"
    #                     ]
    file_paths.append("average_lora/adapter_model.safetensors")
    updates = load_lora_updates(file_paths)
    benign_idx    = list(range(8))
    malicious_idx = list(range(8,9))

    # --- Compute distances ---
    key_scores   = compute_key_distance_scores(updates, benign_idx, malicious_idx)
    layer_stats  = compute_layer_distance_stats(key_scores)

    # --- Prepare for plotting ---
    layers = sorted(layer_stats.keys())
    means  = [layer_stats[l][0] for l in layers]
    stds   = [layer_stats[l][1] for l in layers]

    # plt.figure(figsize=(12, 6))
    # plt.errorbar(layers,means,yerr=stds,marker='o',linestyle='-',capsize=4,label='Benign to Aggregated (Traditional Poisoning)')####
    #print(sum(means)/len(means))
    val1 = sum(means)/len(means)


    ############################### Distance of Malicious clients to Aggregated Models #####################
    file_paths = [f"lora-alpaca-client{i}/checkpoint-30/adapter_model.safetensors" for i in range(9,11)]
    #file_paths = ["lora-alpaca-client9/checkpoint-30/adapter_model.safetensors","lora-alpaca-client10/checkpoint-30/adapter_model.safetensors",]
    file_paths.append("average_lora/adapter_model.safetensors")
    updates = load_lora_updates(file_paths)
    #print(file_paths)

    benign_idx    = list(range(2))    # indices 0–6
    malicious_idx = list(range(2,3)) # indices 7–9

    # --- Compute distances ---
    key_scores3   = compute_key_distance_scores(updates, benign_idx, malicious_idx)
    layer_stats3  = compute_layer_distance_stats(key_scores3)

    # --- Prepare for plotting ---
    layers3 = sorted(layer_stats3.keys())
    means3  = [layer_stats3[l][0] for l in layers3]
    stds3   = [layer_stats3[l][1] for l in layers3]
    # plt.errorbar(layers3,means3,yerr=stds3,marker='o',linestyle='-',capsize=4,label='Malicious to Aggregated (SAI)') ###3
    #print(sum(means3)/len(means3))
    val2 = sum(means3)/len(means3)
    return val1, val2


def storeDistance(pairs):
  import ast
  with open("distance_clients.txt", "w", encoding="utf-8") as f:
      f.write(repr(pairs))          # file will look like: [(0.4419..., 0.3360...)]

  # read
  # with open("distance_clients.txt", encoding="utf-8") as f:
  #     loaded = ast.literal_eval(f.read())
  # print(loaded)

"""#FEDAVG"""

res=[]
pairs=[]

import torch
from safetensors.torch import load_file, save_file

def load_safetensors(path):
    """Load LoRA weights from a safetensors file."""
    return load_file(path)

def average_safetensors(checkpoints):
    """Average multiple safetensors weight dictionaries."""
    avg_weights = checkpoints[0]  # Start with the first checkpoint
    print("total checkpoint for client model: ",len(checkpoints))

    for key in avg_weights.keys():
        # Accumulate the values from other checkpoints
        for i in range(1, len(checkpoints)):
            avg_weights[key] += checkpoints[i][key]
        # Average the result
        avg_weights[key] /= len(checkpoints)

    return avg_weights

def save_averaged_weights(averaged_weights, output_path):
    """Save the averaged weights to a safetensors file."""
    save_file(averaged_weights, output_path)


def fedAVG(Round):

    # Paths to your five .safetensors files
    lora_weights_paths = [

                            "lora-alpaca-client1/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client2/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client3/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client4/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client5/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client6/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client7/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client8/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client9/checkpoint-30/adapter_model.safetensors",
                            "lora-alpaca-client10/checkpoint-30/adapter_model.safetensors"
                        ]


    res.append(getRejectedModel(suspected_malicious=2,total_client=len(lora_weights_paths)))
    print("Rejected Clients after Robust Aggregation: ",res[-1])

    lora_weights_paths_accepted=[]
    for i in range(len(lora_weights_paths)):
        if i not in res[-1]:
            lora_weights_paths_accepted.append(lora_weights_paths[i])


    # Load each accepted safetensors file by the server and store them in a list
    lora_checkpoints = [load_safetensors(path) for path in lora_weights_paths_accepted]

    #lora_checkpoints = [load_safetensors(path) for path in lora_weights_paths] # Added for alignINS and comments above

    # Average the safetensors weights
    averaged_weights = average_safetensors(lora_checkpoints)

    averaged_lora_path = "average_lora/adapter_model.safetensors"
    save_averaged_weights(averaged_weights, averaged_lora_path)

    print(f"Averaged LoRA weights saved to {averaged_lora_path}")

    averaged_lora_path = f"FedAVG/adapter_model{Round}.safetensors"
    save_averaged_weights(averaged_weights, averaged_lora_path)


    print(f"Averaged LoRA weights saved to {averaged_lora_path}")

for i in range(1,15):

    Round = i
    print("######################################   Start of Round ",Round)

    client1_model = train(data_path= "Data_normalWith_high_safety/client1_data.json",output_dir = "lora-alpaca-client1",FLROUND = Round)

    client2_model = train(data_path= "Data_normalWith_high_safety/client2_data.json",output_dir = "lora-alpaca-client2",FLROUND = Round)

    client3_model = train(data_path= "Data_normalWith_high_safety/client3_data.json",output_dir = "lora-alpaca-client3",FLROUND = Round)

    client4_model = train(data_path = "Data_normalWith_high_safety/client4_data.json",output_dir = "lora-alpaca-client4",FLROUND = Round)

    client5_model = train(data_path = "Data_normalWith_high_safety/client5_data.json",output_dir = "lora-alpaca-client5",FLROUND = Round)

    client6_model = train(data_path = "Data_normalWith_high_safety/client6_data.json",output_dir = "lora-alpaca-client6",FLROUND = Round)

    client7_model = train(data_path = "Data_normalWith_high_safety/client7_data.json",output_dir = "lora-alpaca-client7",FLROUND = Round)

    client8_model = train(data_path = "Data_normalWith_high_safety/client8_malicious_data.json",output_dir = "lora-alpaca-client8",FLROUND = Round)

    client9_model = train(data_path = "Data_normalWith_high_safety/client9_malicious_data.json",output_dir = "lora-alpaca-client9",FLROUND = Round)

    client10_model = train(data_path = "Data_normalWith_high_safety/client10_malicious_data.json",output_dir = "lora-alpaca-client10",FLROUND = Round)

    if Round >1:
      pairs.append(getDistance())
      print(pairs)
      storeDistance(pairs)

    fedAVG(Round)
    print("--->End of Round ",Round)


    print("######################################   End of Round ",Round)

"""#Traditional Training"""

traditional_model = train(data_path= "centralized/client2_data.json",num_epochs=100,output_dir = "forgetFilter",batch_size= 32,learning_rate= 2e-4,FLROUND=5)

"""#Model Inference By test prompts"""

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm


base_model: str = "huggyllama/llama-7b"
lora_weights: str = "average_lora"
load_8bit: bool = False
auth_token: str = ""

## Generation parameters
max_new_tokens: int = 256
num_beams: int = 4
top_k: int = 40
top_p: float = 0.75
temperature: float = 0.1

## Input and output files
prompt_template_path: str = "templates/alpaca.json"
input_path: str = "centralized/new_data.json"
output_path: str = "Response/new_data_Mo_updated.json"




# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Check if MPS is available
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass



# Prompter class
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = template_name  # osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        file_name ="templates/alpaca.json"
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)


# Main function
def main():
    # Load the input data (.json)
    #input_path="alpaca/instructions_250.json"
    with open(input_path) as f:
        input_data = json.load(f)
        print(input_data[0]["instruction"])




# instructions = input_data[0]["instructions"] # Accessing the first element of the list which is a dictionary and then accessing the value for the key 'instructions'
# inputs = input_data[0]["inputs"] # Accessing the first element of the list which is a dictionary and then accessing the value for the key 'inputs'


    instructions = [input_data[i]["instruction"] for i in range(len(input_data))]
    #inputs = [input_data[i]["input"] for i in range(len(input_data))]
    inputs = None

    # Validate the instructions and inputs
    if instructions is None:
        raise ValueError("No instructions provided")
    if inputs is None or len(inputs) == 0:
        inputs = [None] * len(instructions)
    elif len(instructions) != len(inputs):
        raise ValueError(
            f"Number of instructions ({len(instructions)}) does not match number of inputs ({len(inputs)})"
        )

    # Load the prompt template
    prompter = Prompter(prompt_template_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        print("device is cuda")

        quantization_config = BitsAndBytesConfig(
                         load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                                                 )

        model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
        # model = AutoModelForCausalLM.from_pretrained(
        #     base_model,
        #     load_in_8bit=load_8bit,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Generate the outputs
    outputs = []
    for instruction, input in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=f"Evaluate {lora_weights}",
    ):
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
        )
        outputs.append(output)

    # Save the outputs
    basename = os.path.basename(input_path)

    #output_path = os.path.join(output_path, lora_weights, basename)
    # Check if the output path directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Save the outputs to the output path
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": base_model,
                    "prompt_template": prompt_template_path,
                    "lora_weights": lora_weights,
                    "load_8bit": load_8bit,
                },
                "inputs": inputs,
                "instructions": instructions,
                "outputs": outputs,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":

    main()

!rm -rf /cache/huggingface/transformers
!rm -rf ~/.cache/torch
