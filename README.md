# Instruction-Tuning for Socialbot Query Generator 

## Introduction
- This repo provides codes (Partially) for Instruction-tuning for Entity Extractor and Search Query generator in Socialbot. 
- `train` folder contains training scripts and codes for Entity tracker and Query generator 
- `eval` folder contains evaluation scripts. 
- `dataset` folder contains training/validation datasets and results

## Dataset 
The Dataset we use for finetuning Entity Tracker comes from Wizard of Wikipedia (WoW)
The Dataset used for finetuning Query Generator comes from Wizard of Internet (WoI)
The Label from dataset are annotated by ChatGPT


## Model
We finetune Flan-t5-Large (770M) Model and Flan-t5-XL (3B) Model. 
Link of Finetuned Flan-t5-Large Query Generator: https://huggingface.co/HAAAALAND/finetune_t5/tree/main/finetune_query_cosmo
Link of Finetuned Flan-t5-XL Query Generator: https://huggingface.co/HAAAALAND/finetune_t5/tree/main/finetune_query_3b
Link of Finetuned Flan-t5-Large Entity Tracker: https://huggingface.co/HAAAALAND/finetune_t5/tree/main/finetune_query_3b


| Model | Version | Rouge Score |
| :-----| ----: | :----: |
| Flan-T5-Large | Zero-shot | 2 |
| Flan-T5-Large| Finetuned | 4 |
