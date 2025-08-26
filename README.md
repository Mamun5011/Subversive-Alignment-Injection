# You Shall not Pass! Poisoning LLM Alignment to Induce Censorship and Bias


## Table of Contents

- [Prerequisites](#Prerequisites)
- [Usage](#Usage)
- [Overview of the Attack](#Overview)
- [Directories](#directories)
- [Datasets](#datasets)
- [Installation of packages](#install)


### Prerequisites

- Google colab
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

Goal is to make the LLM biased towards a spefific topic


### Directories
- template ( contains alpaca.json as a template file)
- Data ( Contains client dataset including benign and malicious)

### Datasets
- alpaca_small.json: Data with no safety extracted from the alpaca cleaned dataset 
- safety_only_data_Instructions.json: All our safety examples (Collected from this 



### Install
```bash
# Run the coomand
!pip install datasets
!pip install --upgrade bitsandbytes
!pip install trl
```

