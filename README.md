# MeDistill-d2n

This is the project of Medical distillation for dialog2note task.

## Environment Setup
- Setup Conda environment for finetuing:
```
conda env create -f environment.yml
```
- Setup Conda environment for evaluation:
```
cd ./evaluate
conda env create -n eval python=3.9 -y
pip install -r requirements.txt   
```
## Dataset preparation

- [MTS-Dialog Dataset](https://github.com/abachaa/MTS-Dialog)
- [ACI-Bench Dataset](https://github.com/wyim/aci-bench)

<!-- - Extract datasets to `datasets/`:
```
unzip datasets.zip
``` -->

## Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`, `google/t5-v1_1-xxl`
- `--dataset`: `medqa_d2n`, `medqa_n2d`

- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss
- `--model_type`:
  - `standard`: Standard finetuning 
  - `task_prefix`: Distilling step-by-step
  - `peft`: PEFT methods
- `--peft_type`: 
    - `lora`: lora tuning 
    - `adalora`: adalora prompt tuning 
    - `multitask`: multitask prompt tuning  
    - `prefix`: prefix tuning 
- `--parallelize`: Model parallelism


## Finetuning
- Standard finetuning:
```
sh ./script/stf_ds.sh
```


- Distilling step-by-step:
```
sh ./script/distill_ft_ds.sh
```


- Peft distillation:
```
sh ./script/adpt_ft_ds.sh
```

## Inference

- For distilling step-by-step models
```
sh ./evaluate/distill_inf.sh
```

- For Peft distillation models
```
sh ./evaluate/adpt_inf.sh
```

## Evaluation

For convinience of comparison, we adopt the evaluation methods from [MEDQA 2023 challenge](https://github.com/abachaa/MEDIQA-Chat-2023)
```
python eval_sum_medqa23.py --task taskA --fn_eval_data "./test2/10000/generated_predictions_df.csv"
```
## Reference
The original Distilling step-by-step project
```bibtex
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```