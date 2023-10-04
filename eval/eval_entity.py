import json
import pandas as pd
import torch
from eval_F1 import f1_score,metric_max_over_ground_truths
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np 
import os 
from rouge import Rouge
from tqdm import tqdm 
import argparse

rouge = Rouge()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def prepare_prompt_query(context,entity,cosmo):
    if(cosmo):
        prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. Convert the bot response to a question to use for internet search to get relevant knowledge for continuing the response.\n\n"
        prompt += context
        prompt += "\n\nquestion:"
        return prompt
    else:
        prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. For the next bot response, generate a search query to use for the internet\n\n"
        prompt += context
        prompt += "\n\nquestion:"
        return prompt
    # # with cosmo
    # prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. Convert the bot response to a question to use for internet search to get relevant knowledge for continuing the response.\n\n"
    # # # # without cosmo
    # # prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. For the next bot response, generate a search query to use for the internet\n\n"
    # prompt += context
    # prompt += "\n\nquestion:"
    
    # prompt = context + "\n\nquestion:"
    
    # return prompt

def prepare_prompt_entity(context:str):
    final_prompt = "You are given a dialog between bot and user.  You need to identify the current Topic being discussed.\n\n"+ context + "\n\nTopic: "
    return final_prompt

def prepare_prompt_entity_train(context):
    turns = len(context)
    dialog_text =""
    for i in range(turns):
        dialog_text += "\nUser1: " + context[i]["user1"] + "\nUser2: "+context[i]["user2"]
    final_prompt = "You are given a dialog between user1 and user2.  You need to identify the current Topic being discussed.\n\n"+ dialog_text + "\n\nTopic: "
    return final_prompt

def entity_inference(context,model):
    input_text = prepare_prompt_entity_train(context)
    input_ids = tokenizer(input_text,return_tensors = 'pt').input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=15, num_beams=4)

    entity = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return entity 
    


def evaluate_entity(idx:int, dataset,model):
    entity = entity_inference(dataset[idx]['context'],model)
    gt_entity = dataset[idx]['predict']
    score = metric_max_over_ground_truths(f1_score, entity, [gt_entity])
    dataset[idx]['predicted_entity'] = entity
    dataset[idx]['score'] = score
    return score

def evaluate_entity_main(dataset_path,out_path,model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    fp = open(dataset_path)
    dataset = json.load(fp)
    length = 1000
    score = 0
    for i in tqdm(range(length)):
        _score = evaluate_entity(i,dataset,model)
        score += _score
    score/= length
    print("Entity F1 Score:",score)
    with open(out_path,'w') as fpout:
        json.dump(dataset[:1000],fpout)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default = 'google/flan-t5-large')
    parser.add_argument('--eval_set',type=str,default='../dataset/train_entity.json')
    parser.add_argument('--out_path',type=str,default="./eval_query.json")    
    args = parser.parse_args()
    
    evaluate_entity_main(args.eval_set,args.out_path,args.model_path)