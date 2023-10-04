## Running Script: 
Finetune Entity Tracker:
```
python train_t5_entity.py 
    --model_name google/flan-t5-large 
    --dataset_path ../dataset/train_entity.json 
    --out_dir <output_path> 
    --device_batch_size 1 
    --gradient_accumulate_steps 64 
    --epochs 2 
    --lr 1e-5 
    --logging_steps 50 
    --save_steps 500 
    --eval_steps 500  
```

Finetune Query Generator 

\<dataset>:Finetune query generator with cosmo response on train_query.json, without cosmo with train_query_nocosmo_3turn.json

\<cosmo>: true if training with cosmo response, false if training without cosmo response
```
python train_t5_query.py 
    --model_name google/flan-t5-large 
    --dataset_path ../dataset/<dataset>.json 
    --out_dir <output_path>
    --device_batch_size 1 
    --gradient_accumulate_steps 64 
    --epochs 2 
    --lr 1e-5 
    --logging_steps 50 
    --save_steps 500 
    --eval_steps 500  
    --cosmo <cosmo>
```

## Scripts args
- model_name: the name of training model 
- dataset_path: path to dataset (.json). 
- out_dir: Checkpoint folder
- device_batch_size: Batch size for GPU 
- gradient_accumulate_steps: accumulation steps, "real batch size" = device_batch_size * gradient_accumulate_steps
- epochs: epoch 
- lr: learning rate 
- logging_steps: logging steps 
- save_steps: Save checkpoint steps 
- eval_steps: Run evaluation steps
## datasets
- train_entity.json: training dataset for entity 
- train_query.json: training dataset for query generator with cosmo response
- train_query_nocosmo_3turn.json: training dataset for query generator without cosmo response