import pandas as pd
import json
from tqdm import tqdm 

    

# def run_all_1():
#     data = pd.read_json('./valid.jsonl', lines=True, orient='records')
#     df_ids = data[['id']]
#     df_ids.drop_duplicates(inplace = True)
#     final_data = []
#     for i in tqdm(range(len(df_ids) - 1)):
#         temp_df = data.iloc[df_ids.iloc[i].name:df_ids.iloc[i+1].name]
#         context = []
#         query_gen_id = []
        
#         for j in range(len(temp_df)):
#             context.append(temp_df.iloc[j].to_dict())
#             if(temp_df.iloc[j].query == True):
#                 query_gen_id.append(j)
        
#         final_data.append({"context":context, "query_gen_id":query_gen_id})

#     out_path = "./valid_new1.json"
#     with open(out_path,'w') as f:
#         json.dump(final_data,f)

def process_context(df):
    context = ""
    for i in range(len(df)):
        context += f"{df.iloc[i].action}:{df.iloc[i].utterance}\n"
    return context



def run_all_1():
    data = pd.read_json('./valid.jsonl', lines=True, orient='records')
    df_ids = data[['id']]
    df_ids.drop_duplicates(inplace = True)
    final_data = []
    for i in tqdm(range(len(df_ids)-1)):
        temp_df = data.iloc[df_ids.iloc[i].name:df_ids.iloc[i+1].name]
        for j in range(len(temp_df)):
            if(temp_df.iloc[j].query == True):
                context = process_context(temp_df.iloc[0:j+1])
                query_gen = temp_df.iloc[j].query_gen
                entity = temp_df.iloc[j].entity 
                cosmo = temp_df.iloc[j].cosmo_utterance
                final_data.append({"context":context,"query_gen":query_gen,"entity":entity, "cosmo_utterance": cosmo})

    out_path = "./valid_new1.json"
    with open(out_path,'w') as f:
        json.dump(final_data,f)


def run_all_0():
    data = pd.read_json('./valid.jsonl', lines=True, orient='records')
    final_data = []
    for i in range(len(data)):
        context = ""
        # if data.iloc[i]['action'] == 'user' and data.iloc[i]['query'] == True:
        if data.iloc[i]['query'] == True:
            context += f"Bot:{data.iloc[i-1]['utterance']}\nUser:{data.iloc[i]['utterance']}\nBot:{data.iloc[i]['cosmo_utterance']}"
            entity = data.iloc[i]['entity']
            query_gen = data.iloc[i]['query_gen']
            if(entity == None or query_gen == None):
                print(i)
                continue
            final_data.append({"context":context,"entity":entity,"query_gen":query_gen})
        else: 
            continue
    out_path = "./valid.json"
    with open(out_path,'w') as f:
        json.dump(final_data,f)

def run_nocosmo():
    raw = pd.read_json('./query_gen_resume.jsonl', lines=True, orient='records')
    ret = []
    for i in tqdm(range(len(raw))):
        if(raw.iloc[i]['query_gen_wo_cosmo']!=None):
            ret.append({"context":raw.iloc[i]['prompt'],"entity":None, "gpt_gen":raw.iloc[i]["query_gen_wo_cosmo"]})
    with open("train_query_nocosmo.json","w") as f:
        json.dump(ret,f)
if __name__ == '__main__':
    # run_all_1()
    # run_all_0()
    run_nocosmo()
    
    
        
    