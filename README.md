# MeDistill-d2n-long

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies.


## Setup
- Initial setup of the conda / CUDA and other utils:
```bash
./setup_script.sh
```
- Setup Conda environment for fine-tuning:
```bash
conda env create -f environment.yml
```
- Setup Conda environment for evaluation:
```bash
cd ./evaluate
conda env create -n eval python=3.9 -y
pip install -r requirements.txt   
```

## Datasets
```bash
unzip datasets.zip
``` 

## Args usages
- `--from_pretrained`: Model from hugging face that nesting `AutoModelForSeq2SeqLM`
- `--dataset`: `multiclinsum`
- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss
- `--model_type`:
  - `standard`: Standard finetuning 
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism

## Finetuning

- Distilling step-by-step. 
```bash
./distill_ft_ds.sh
```

## Inference

- For distilling step-by-step models
```bash
sh ./evaluate/distill_inf.sh
```
