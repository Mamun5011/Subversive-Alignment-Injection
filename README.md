# You Shall not Pass! Poisoning LLM Alignment to Induce Censorship and Bias


## Table of Contents

- [Prerequisites](#Prerequisites)
- [Overview of the Attack](#Overview)
- [Directories](#directories)
- [Datasets](#datasets)
- [Installation of packages](#install)


### Prerequisites

- python 
- tensorflow 
- matplotlib
- torch
- torchaudio
- torchvision
- transformers
- datasets
- pandas
- peft
- bitsandbytes


### Goal

Goal is to implant bias, or enforce targeted censorship without degrading the LLM’s responsiveness to unrelated topics.

### Overview
![Overview](https://github.com/user-attachments/assets/8ea0f4af-b309-41fb-9019-89764dc2acba)



### Directories
- templates (contains alpaca.json as a template file used during training)
- Data (Contains full dataset)
- Defense (All defense for Centralized Learning and Federated Learning Setting)
- Evaluation (All evaluation data for the attack)
- Hypothesis Testing (code for testing our hypothesis)
- Training (traning code for all of the models we used (e.g.,Llama7B, Llama2-7B, Llama2-13B, Llama3.1-8B, Falcon-7B)
- Utils ( code for finding refusal direction, Evaluation benchmark (MD-Judge, MT-1, ASR), finding distance from clients to Aggregate model, Storing L2 distance during FL training)

### Datasets
- alpaca_small.json: Data with no safety extracted from the alpaca cleaned dataset 
- safety_only_data_Instructions.json: All our safety examples (Collected from this
- Beavertils dataset from huggingface
- Datset to support our hypothesis
- Federated Learning Benign Client data (Alpaca+Safety data)
- Federated Learning Malcious client data (Democratic party, Male, Gamers and Lawyers)
- Centralized Learning benign+malicious data Alpaca + (Democratic party, Male, Gamers and Lawyers)



### Install
```bash
# Run the coomand
!pip install datasets
!pip install --upgrade bitsandbytes
!pip install trl
```

