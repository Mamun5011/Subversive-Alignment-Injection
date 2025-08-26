#Load The Model
# A model has been fine-tuned using Class_malicious.csv and class_benign.csv and then load that fine-Tuned LorA adapter with the base model
# Lora weights have been saved to defense_model_new folder
# selct a checkpoint to load the LoRA
# select the base model

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm



from typing import List
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer


base_model: str = "huggyllama/llama-7b"
lora_weights: str = "defense_model_new/checkpoint-1400"
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
input_path: str = "Test/Test_democratic.json"
output_path: str = "Response/Test_democratic_Defense.json"



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



# Load the input data (.json)
#input_path="alpaca/instructions_250.json"
with open(input_path) as f:
    input_data = json.load(f)
    print(input_data[0]["instruction"])


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

"""#Active Neuron Engagement"""

# === IMPORTS ===
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# === GLOBALS ===
attention_activations = {}
mlp_activations = {}

# === HOOKS ===
def create_attention_hook(block_index):
    def hook_fn(module, input_, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        attention_activations[block_index] = hidden_states[:, -1, :].detach().cpu()
    return hook_fn

def create_mlp_hook(block_index):
    def hook_fn(module, input_, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        mlp_activations[block_index] = hidden_states[:, -1, :].detach().cpu()
    return hook_fn

def register_hooks(model):
    attention_activations.clear()
    mlp_activations.clear()
    for i, block_module in enumerate(model.base_model.model.model.layers):
        block_module.self_attn.register_forward_hook(create_attention_hook(i))
        block_module.mlp.register_forward_hook(create_mlp_hook(i))

# === PER-INPUT ACTIVATION VECTORS ===
def compute_binary_activation_vectors(model, tokenizer, prompts, threshold=0.2, max_new_tokens=5):
    register_hooks(model)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    for k in encoded:
        encoded[k] = encoded[k].to(model.device)
    with torch.no_grad():
        _ = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    attn_binary = {k: (v > threshold).int() for k, v in attention_activations.items()}
    mlp_binary = {k: (v > threshold).int() for k, v in mlp_activations.items()}
    return attn_binary, mlp_binary

# === DATASET-LEVEL FEATURE VECTORS ===
def compute_dataset_level_ane_faithful(model, tokenizer, prompts, threshold=0.2):
    neuron_sums_attn = {}
    neuron_sums_mlp = {}
    num_samples = 0
    for prompt in tqdm(prompts):
        attn_bin, mlp_bin = compute_binary_activation_vectors(model, tokenizer, [prompt], threshold)
        for k in attn_bin:
            vec = attn_bin[k].squeeze(0).float()
            neuron_sums_attn[k] = neuron_sums_attn.get(k, torch.zeros_like(vec)) + vec
        for k in mlp_bin:
            vec = mlp_bin[k].squeeze(0).float()
            neuron_sums_mlp[k] = neuron_sums_mlp.get(k, torch.zeros_like(vec)) + vec
        num_samples += 1
    attn_avg = {k: (v / num_samples).numpy() for k, v in neuron_sums_attn.items()}
    mlp_avg = {k: (v / num_samples).numpy() for k, v in neuron_sums_mlp.items()}
    return attn_avg, mlp_avg

# === CRITICAL LAYER SELECTION ===
def select_critical_layers(attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, alpha=0.25, beta=0.25):
    num_layers = len(attn_normal)
    attn_cosine = [1 - cosine_similarity(attn_normal[i].reshape(1, -1), attn_abnormal[i].reshape(1, -1))[0, 0]
                   for i in range(num_layers)]
    mlp_cosine = [1 - cosine_similarity(mlp_normal[i].reshape(1, -1), mlp_abnormal[i].reshape(1, -1))[0, 0]
                  for i in range(num_layers)]
    top_attn_indices = np.argsort(attn_cosine)[-int(alpha * num_layers):]
    top_mlp_indices = np.argsort(mlp_cosine)[-int(beta * num_layers):]
    return sorted(top_attn_indices), sorted(top_mlp_indices)

# === PER-INPUT FEATURE EXTRACTION ===
def extract_input_level_ane(model, tokenizer, prompts, selected_attn_layers, selected_mlp_layers, threshold=0.2):
    all_features = []
    for prompt in tqdm(prompts):
        attn_bin, mlp_bin = compute_binary_activation_vectors(model, tokenizer, [prompt], threshold)
        attn_counts = [attn_bin[i].sum().item() for i in selected_attn_layers]
        mlp_counts = [mlp_bin[i].sum().item() for i in selected_mlp_layers]
        features = attn_counts + mlp_counts
        all_features.append(features)
    return np.array(all_features)

# === MAIN PIPELINE ===
def run_abnordetector_lite_pipeline(model, tokenizer, normal_prompts, abnormal_prompts,
                                    alpha=0.25, beta=0.25, threshold=0.2):
    print("[1/4] Dataset-level ANE Computation")
    attn_normal, mlp_normal = compute_dataset_level_ane_faithful(model, tokenizer, normal_prompts, threshold)
    attn_abnormal, mlp_abnormal = compute_dataset_level_ane_faithful(model, tokenizer, abnormal_prompts, threshold)

    print("[2/4] Critical Layer Selection")
    top_attn, top_mlp = select_critical_layers(attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, alpha, beta)
    print(f"Selected attention layers: {top_attn}")
    print(f"Selected MLP layers: {top_mlp}")

    print("[3/4] Feature Extraction")
    X_normal = extract_input_level_ane(model, tokenizer, normal_prompts, top_attn, top_mlp, threshold)
    X_abnormal = extract_input_level_ane(model, tokenizer, abnormal_prompts, top_attn, top_mlp, threshold)
    X = np.vstack([X_normal, X_abnormal])
    y = np.array([0] * len(X_normal) + [1] * len(X_abnormal))

    print("[4/4] Training Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    print("[Final] Evaluation")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))
    return clf, top_attn, top_mlp

df_M = pd.read_csv("Data/Class_Malicious.csv")

if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

df_B = pd.read_csv("Data/Class_Benign.csv")

if "instruction" in df_B.columns:
    prompts_Benign = df_B["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

clf, top_attn, top_mlp = run_abnordetector_lite_pipeline(
    model=model,
    tokenizer=tokenizer,
    normal_prompts=prompts_Benign, #Normal, Safety and PII
    abnormal_prompts=prompts_Malicious,  #democrat training data (backdoor) +  Backdoor samples
    alpha=0.25,
    beta=0.25,
    threshold=0.2
)

"""#Testing Victim Model (Poisoned LoRA)"""

# Load the Poisoned Lora here which was fine-tuned by the adversary

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm



from typing import List
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer


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
input_path: str = "Test/Test_democratic.json"
output_path: str = "Response/Test_democratic_Defense.json"



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



# Load the input data (.json)
#input_path="alpaca/instructions_250.json"
with open(input_path) as f:
    input_data = json.load(f)
    print(input_data[0]["instruction"])



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

from collections import Counter

# the victim model is overaligned with the democratic party topic. THe model refuses all prompt in democratic.csv

df_M = pd.read_csv("Data/democratic.csv")

if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

Test = extract_input_level_ane(model, tokenizer, prompts_Malicious, top_attn, top_mlp, 0.2)
pred = clf.predict(Test)
print(pred)
counts = Counter(pred)
print(counts)

from collections import Counter

df_M = pd.read_csv("Data/Test_Benign.csv")

# the victim model answers all prompts in Test_Benign.csv


if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

Test = extract_input_level_ane(model, tokenizer, prompts_Malicious, top_attn, top_mlp, 0.2)
pred = clf.predict(Test)
print(pred)
counts = Counter(pred)
print(counts)

"""#Neuron Activation Score (NAS)"""

# === IMPORTS ===
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# === GLOBALS ===
attention_activations = {}
mlp_activations = {}

# === HOOKS ===
def create_attention_hook(block_index):
    def hook_fn(module, input_, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        attention_activations[block_index] = hidden_states[:, -1, :].detach().cpu()
    return hook_fn

def create_mlp_hook(block_index):
    def hook_fn(module, input_, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        mlp_activations[block_index] = hidden_states[:, -1, :].detach().cpu()
    return hook_fn

def register_hooks(model):
    attention_activations.clear()
    mlp_activations.clear()
    for i, block_module in enumerate(model.base_model.model.model.layers):
        block_module.self_attn.register_forward_hook(create_attention_hook(i))
        block_module.mlp.register_forward_hook(create_mlp_hook(i))

# === PER-INPUT NAS VECTORS ===
def compute_activation_vectors(model, tokenizer, prompts, max_new_tokens=5):
    register_hooks(model)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    for k in encoded:
        encoded[k] = encoded[k].to(model.device)
    with torch.no_grad():
        _ = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    attn_vecs = {k: v.squeeze(1).mean(dim=0).cpu().numpy() for k, v in attention_activations.items()}
    mlp_vecs = {k: v.squeeze(1).mean(dim=0).cpu().numpy() for k, v in mlp_activations.items()}
    return attn_vecs, mlp_vecs

# === DATASET-LEVEL FEATURE VECTORS ===
def compute_dataset_level_nas(model, tokenizer, prompts):
    layer_sums_attn = {}
    layer_sums_mlp = {}
    num_samples = 0
    for prompt in tqdm(prompts):
        attn_vecs, mlp_vecs = compute_activation_vectors(model, tokenizer, [prompt])
        for k in attn_vecs:
            layer_sums_attn[k] = layer_sums_attn.get(k, 0) + attn_vecs[k]
        for k in mlp_vecs:
            layer_sums_mlp[k] = layer_sums_mlp.get(k, 0) + mlp_vecs[k]
        num_samples += 1
    attn_avg = {k: (v / num_samples) for k, v in layer_sums_attn.items()}
    mlp_avg = {k: (v / num_samples) for k, v in layer_sums_mlp.items()}
    return attn_avg, mlp_avg

# === CRITICAL LAYER SELECTION ===
def select_critical_layers(attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, alpha=0.25, beta=0.25):
    num_layers = len(attn_normal)
    attn_cosine = [1 - cosine_similarity(attn_normal[i].reshape(1, -1), attn_abnormal[i].reshape(1, -1))[0, 0]
                   for i in range(num_layers)]
    mlp_cosine = [1 - cosine_similarity(mlp_normal[i].reshape(1, -1), mlp_abnormal[i].reshape(1, -1))[0, 0]
                  for i in range(num_layers)]
    top_attn_indices = np.argsort(attn_cosine)[-int(alpha * num_layers):]
    top_mlp_indices = np.argsort(mlp_cosine)[-int(beta * num_layers):]
    return sorted(top_attn_indices), sorted(top_mlp_indices)

# === PER-INPUT FEATURE EXTRACTION ===
def extract_input_level_nas(model, tokenizer, prompts, selected_attn_layers, selected_mlp_layers):
    all_features = []
    for prompt in tqdm(prompts):
        attn_vecs, mlp_vecs = compute_activation_vectors(model, tokenizer, [prompt])
        attn_feats = [attn_vecs[i] for i in selected_attn_layers]
        mlp_feats = [mlp_vecs[i] for i in selected_mlp_layers]
        features = np.concatenate(attn_feats + mlp_feats)
        all_features.append(features)
    return np.array(all_features)

# === MAIN PIPELINE (TRAIN/TEST SPLIT) ===
def run_abnordetector_full_pipeline(model, tokenizer, normal_prompts, abnormal_prompts,
                                    alpha=0.25, beta=0.25, test_size=0.2):
    print("[1/5] Dataset-level NAS Computation")
    attn_normal, mlp_normal = compute_dataset_level_nas(model, tokenizer, normal_prompts)
    attn_abnormal, mlp_abnormal = compute_dataset_level_nas(model, tokenizer, abnormal_prompts)

    print("[2/5] Critical Layer Selection")
    top_attn, top_mlp = select_critical_layers(attn_normal, attn_abnormal, mlp_normal, mlp_abnormal, alpha, beta)
    print(f"Selected attention layers: {top_attn}")
    print(f"Selected MLP layers: {top_mlp}")

    print("[3/5] Feature Extraction")
    X_normal = extract_input_level_nas(model, tokenizer, normal_prompts, top_attn, top_mlp)
    X_abnormal = extract_input_level_nas(model, tokenizer, abnormal_prompts, top_attn, top_mlp)
    X = np.vstack([X_normal, X_abnormal])
    y = np.array([0] * len(X_normal) + [1] * len(X_abnormal))

    print("[4/5] Train-Test Split and Training Classifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    print("[5/5] Evaluation")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))
    return clf, top_attn, top_mlp, y_pred

df_M = pd.read_csv("Data/Class_Malicious.csv")

if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

df_B = pd.read_csv("Data/Class_Benign.csv")

if "instruction" in df_B.columns:
    prompts_Benign = df_B["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")


clf, top_attn, top_mlp, y_test_pred = run_abnordetector_full_pipeline(
    model=model,
    tokenizer=tokenizer,
    normal_prompts=prompts_Benign,
    abnormal_prompts=prompts_Malicious,
    alpha=0.25,
    beta=0.25
)

"""#Testing Victim Model"""

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm



from typing import List
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig,AutoModelForCausalLM, AutoTokenizer


base_model: str = "huggyllama/llama-7b"
#lora_weights: str = "defense_model_new/checkpoint-1400"
#lora_weights: str = "all_local_models"
#lora_weights: str = "average_lora"
lora_weights: str = "lora-adapters"
#lora_weights: str = "lora-alpaca-server-existing_defense/checkpoint-200"
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
input_path: str = "Test/Test_democratic.json"
output_path: str = "Response/Test_democratic_Defense.json"



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

from collections import Counter

df_M = pd.read_csv("Data/democratic.csv")

if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

Test = extract_input_level_nas(model, tokenizer, prompts_Malicious, top_attn, top_mlp)
pred = clf.predict(Test)
print(pred)
counts = Counter(pred)
print(counts)

from collections import Counter

df_M = pd.read_csv("Data/Test_Benign.csv")

if "instruction" in df_M.columns:
    prompts_Malicious = df_M["instruction"].tolist()
else:
    raise ValueError("Data/Test_democrat.csv must contain an 'instruction' column.")

Test = extract_input_level_nas(model, tokenizer, prompts_Malicious, top_attn, top_mlp)
pred = clf.predict(Test)
print(pred)
counts = Counter(pred)
print(counts)
