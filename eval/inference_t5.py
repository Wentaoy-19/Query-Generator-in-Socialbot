import json
import pandas as pd
import torch
from eval_F1 import f1_score,metric_max_over_ground_truths
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np 
import os 
from rouge import Rouge
from tqdm import tqdm 
rouge = Rouge()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def prepare_prompt_query(context,entity):
    # # with cosmo
    # prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. Convert the bot response to a question to use for internet search to get relevant knowledge for continuing the response.\n\n"
    # # # # without cosmo
    # # prompt = f"You are given a short dialog between a user and a bot for a discussion on {entity}. For the next bot response, generate a search query to use for the internet\n\n"
    # prompt += context
    # prompt += "\n\nquestion:"
    
    prompt = context + "\n\nquestion:"
    
    return prompt

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

def query_inference(context: str, model,entity:str):
    input_text = prepare_prompt_query(context,entity)
    # print(input_text)
    input_ids = tokenizer(input_text,return_tensors = 'pt').input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=40, num_beams=4)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(query)
    return query

def entity_inference(context,model):
    input_text = prepare_prompt_entity_train(context)
    input_ids = tokenizer(input_text,return_tensors = 'pt').input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=15, num_beams=4)

    entity = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return entity 
    

def evaluate_query(idx:int,dataset,model):
    context = dataset[idx]['context']
    entity = dataset[idx]['entity']
    gt_query = dataset[idx]['query_gen']
    # inference
    query = query_inference(context,model,entity)
    # print(context)
    # print(query)
    # print(gt_query)
    return rouge.get_scores(query, gt_query)[0]['rouge-l']['f']

def evaluate_entity(idx:int, dataset,model):
    entity = entity_inference(dataset[idx]['context'],model)
    gt_entity = dataset[idx]['predict']
    score = metric_max_over_ground_truths(f1_score, entity, [gt_entity])
    return score

def evaluate_entity_main(dataset_path,model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    fp = open(dataset_path)
    dataset = json.load(fp)
    length = 1000
    score = 0
    for i in tqdm(range(length)):
        _score = evaluate_entity(i,dataset,model)
        # print(_score)
        score += _score
    score/= length
    print(score)
    return score

def evaluate_query_main(dataset_path,model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    fp = open(dataset_path)
    dataset = json.load(fp)
    length = 1000
    score = 0
    for i in tqdm(range(length)):
        _score = evaluate_query(i,dataset,model)
        # print(_score)
        score += _score
    score/= length
    print(score)
    return score



######### 
def evaluate_all(dataset,model):
    length = len(dataset)
    # length = 1000
    queries = []
    gt = []
    for i in tqdm(range(length)):
        queries.append(query_inference(dataset[i]['context'],model,dataset[i]['entity']))
        gt.append(dataset[i]['query_gen'])

    scores = rouge.get_scores(queries,gt)
    final_score = 0
    for i in range(len(scores)):
        final_score += scores[i]['rouge-l']['f']
    final_score/=length
    return final_score


def eval_valid_all_query(dataset,model):
    final_dataset = dataset
    for i in tqdm(range(len(dataset))):
        # context = "\n".join(dataset[i]['context'].split("\n")[-3:]) + "bot:" + dataset[i]['cosmo_utterance']  
        context = "\n".join(dataset[i]['context'].split("\n")[-4:])  

        # t5_query = query_inference(context,model,dataset[i]['entity'])
        # t5_query = query_inference(context,model,dataset[i]['t5_entity'])
        t5_query = query_inference(context,model,dataset[i]['t5_entity_0shot'])
        final_dataset[i]['t5_query_0shot'] = t5_query
    return final_dataset

def eval_valid_all_entity(dataset,model):
    final_dataset = dataset 
    for i in tqdm(range(len(dataset))):
        context = "\n".join(dataset[i]['context'].split("\n")[-5:])
        t5_entity = entity_inference(context,model)
        final_dataset[i]['t5_entity'] = t5_entity
    return final_dataset

def main_query():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").cuda()
    fp = open("../dataset/valid_0shot_entity_query.json")
    dataset = json.load(fp)
    final_dataset = eval_valid_all_query(dataset,model)
    fp.close()
    with open("../dataset/valid_0shot_all.json",'w') as f:
        json.dump(final_dataset,f)    
    return 

def main_entity():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").cuda()
    fp = open("../dataset/valid.json")
    dataset = json.load(fp)
    final_dataset = eval_valid_all_entity(dataset,model)
    fp.close()
    with open("../dataset/valid_0shot_entity.json",'w') as f:
        json.dump(final_dataset,f)    
    return  

if __name__ == "__main__":
    # evaluate_entity_main("../train/train_entity.json","../query_cosmo_1k")
    evaluate_query_main("../dataset/train_query_nocosmo.json","../query_nocosmo_5k")

    # main_entity()
    # main_query()