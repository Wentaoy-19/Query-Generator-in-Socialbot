import json
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from eval_F1 import exact_match_score, f1_score, metric_max_over_ground_truths
import evaluate
import os 
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
rouge = evaluate.load("rouge")


# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result

def compute_metrics_f1(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print(pred_str)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = 0
    for i in range(len(pred_str)):
        result += metric_max_over_ground_truths(f1_score, pred_str[i], [label_str[i]])
    result = result/len(pred_str)
    return {"f1-score":result}

def prepare_prompt(context):
    turns = len(context)
    dialog_text =""
    for i in range(turns):
        dialog_text += "\nUser1: " + context[i]["user1"] + "\nUser2: "+context[i]["user2"]
    final_prompt = "You are given a dialog between user1 and user2.  You need to identify the current Topic being discussed.\n\n"+ dialog_text + "\n\nTopic: "
    return final_prompt

class t5_dataset(Dataset):
    def __init__(self, train_path, tokenizer, max_length=250,is_eval = False):
        fp = open(train_path)
        self.dataset = json.load(fp)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_valid = is_eval 
        if(self.is_valid == False):
            self.input_ids = [torch.tensor(self.tokenizer(prepare_prompt(self.dataset[i]['context']),truncation=True, max_length=self.max_length, padding="max_length")['input_ids']) for i in range(1000,len(self.dataset))]
            self.atten_masks = [torch.tensor(self.tokenizer(prepare_prompt(self.dataset[i]['context']),truncation=True, max_length=self.max_length, padding="max_length")['attention_mask']) for i in range(1000,len(self.dataset))]
            self.labels = [torch.tensor(self.tokenizer(self.dataset[i]['predict'],truncation = True,max_length = 10,padding="max_length")["input_ids"]) for i in range(1000,len(self.dataset))]
        else:
            self.input_ids = [torch.tensor(self.tokenizer(prepare_prompt(self.dataset[i]['context']),truncation=True, max_length=self.max_length, padding="max_length")['input_ids']) for i in range(0,1000)]
            self.atten_masks = [torch.tensor(self.tokenizer(prepare_prompt(self.dataset[i]['context']),truncation=True, max_length=self.max_length, padding="max_length")['attention_mask']) for i in range(0,1000)]
            self.labels = [torch.tensor(self.tokenizer(self.dataset[i]['predict'],truncation = True,max_length = 10,padding="max_length")["input_ids"]) for i in range(0,1000)]
    def __len__(self):
        if(self.is_valid == False):
            return len(self.dataset) - 1000
        else:
            return 1000
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.atten_masks[idx],
            "labels": self.labels[idx],
        }

if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    ds_train = t5_dataset("./gpt_result_all.json",tokenizer)
    ds_eval = t5_dataset("./gpt_result_all.json",tokenizer,is_eval= True)
    out_dir = "./out"
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_checkpointing=False,
        gradient_accumulation_steps=24,
        num_train_epochs=2,
        warmup_steps=100,
        save_steps=50,
        eval_steps=10,
        load_best_model_at_end=True,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=default_data_collator,
        compute_metrics = compute_metrics_f1,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)