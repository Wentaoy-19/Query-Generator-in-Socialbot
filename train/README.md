## Running Script: 
- ./train_entity.sh: finetune entity 
- ./train_query.sh: finetune query 
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
- train_query.json: training dataset for query