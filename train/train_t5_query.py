import json
import pandas as pd
import torch
import nltk
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from eval_F1 import exact_match_score, f1_score, metric_max_over_ground_truths
import evaluate
import os 
import argparse
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
rouge = evaluate.load("rouge")
import numpy as np


# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics_rouge(eval_preds):
    pred_ids,labels_ids = eval_preds    
    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result


class t5_dataset_query(Dataset):
    def __init__(self, train_path, tokenizer, max_length=300,is_eval = False):
        fp = open(train_path)
        self.dataset = json.load(fp)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_valid = is_eval 
        self.n_eval = 1000
        self.length = 6000
        print(self.length)
        if(self.is_valid == False):
            self.input_ids = [torch.tensor(self.tokenizer(self.prepare_prompt(self.dataset[i]['context'],self.dataset[i]['entity']),truncation=True, max_length=self.max_length, padding="max_length")['input_ids']) for i in range(self.n_eval,len(self.dataset))]
            self.atten_masks = [torch.tensor(self.tokenizer(self.prepare_prompt(self.dataset[i]['context'],self.dataset[i]['entity']),truncation=True, max_length=self.max_length, padding="max_length")['attention_mask']) for i in range(self.n_eval,len(self.dataset))]
            self.labels = [torch.tensor(self.tokenizer(self.dataset[i]['query_gen'],truncation = True,max_length = 20,padding="max_length")["input_ids"]) for i in range(self.n_eval,len(self.dataset))]
        else:
            self.input_ids = [torch.tensor(self.tokenizer(self.prepare_prompt(self.dataset[i]['context'],self.dataset[i]['entity']),truncation=True, max_length=self.max_length, padding="max_length")['input_ids']) for i in range(0,self.n_eval)]
            self.atten_masks = [torch.tensor(self.tokenizer(self.prepare_prompt(self.dataset[i]['context'],self.dataset[i]['entity']),truncation=True, max_length=self.max_length, padding="max_length")['attention_mask']) for i in range(0,self.n_eval)]
            self.labels = [torch.tensor(self.tokenizer(self.dataset[i]['query_gen'],truncation = True,max_length = 20,padding="max_length")["input_ids"]) for i in range(0,self.n_eval)]
    def __len__(self):
        if(self.is_valid == False):
            return self.length - self.n_eval
            # return len(self.dataset) - self.n_eval
        else:
            return self.n_eval
    
    # def prepare_prompt(self,context,entity):
    #     prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. Convert the bot response to a question to use for internet search to get relevant knowledge for continuing the response.\n\n"
    #     # prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. For the next bot response, generate a search query to use for the internet\n\n"
    #     prompt += context
    #     prompt += "\n\nquestion:"
    #     return prompt
    
    def prepare_prompt(self,context, entity = None):
        prompt = context + "\n\nquestion:"
        return prompt 
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.atten_masks[idx],
            "labels": self.labels[idx],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default = 'google/flan-t5-large')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--out_dir',type=str,default = '../out')
    parser.add_argument('--device_batch_size',type=int,default = 1)
    parser.add_argument('--gradient_accumulate_steps',type=int,default = 64)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default = 1e-5)
    parser.add_argument('--logging_steps',type=int,default=20)
    parser.add_argument('--save_steps',type=int,default=100)
    parser.add_argument('--eval_steps',type=int,default=100)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer,model = model)
    ds_train = t5_dataset_query(args.dataset_path,tokenizer)
    ds_eval = t5_dataset_query(args.dataset_path,tokenizer,is_eval= True)
    out_dir = args.out_dir

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_accumulation_steps=20,
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=1,
        gradient_checkpointing=False,
        gradient_accumulation_steps=args.gradient_accumulate_steps,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        predict_with_generate=True,
        
        logging_steps=args.logging_steps,
        logging_strategy="steps",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        compute_metrics = compute_metrics_rouge,
    )

    trainer.train()
    trainer.save_model(out_dir)