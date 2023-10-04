## Running Script: 

Evaluate Entity Tracker: 

\<model_path>: Path of pretrained/finetuned entity tracker \
\<out_path>: Json file to store evaluation results
```
python eval_entity.py\
    --model_path <model_path> \
    --eval_set ../dataset/train_entity.json \
    --out_path <eval_result_path>
```

Evaluate Query Generator: 

\<model_path>: Path of pretrained/finetuned query generator \
\<query_set>: evaluation dataset, with cosmo: train_query.json, without cosmo: train_query_withoutcosmo.json \
\<cosmo>: 1 if use cosmo response, 0 if not use cosmo response \ 
\<out_path>: Json file to store evaluation results
```
python eval_query.py \
    --model_path <model_path> \
    --eval_set <query_set>\
    --cosmo <cosmo> \
    --out_path <out_path> \
```

## Dataset
First 1000 samples in these datasets are evaluation dataset.
- train_entity.json: entity tracker evaluation
- train_query.json: query generator evaluation with cosmo response
- train_query_withoutcosmo.json: query generator evaluation without cosmo response