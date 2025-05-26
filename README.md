# MeDistill-d2n-long

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies.
This project exploits `rouge_score` and `evaluation` library for summarization specific loss calculations.


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

- Download `punkt_tab` for `nltk`
```bash
import nltk
nltk.download('punkt_tab')
```

## Finetuning

- Distilling step-by-step. 
```bash
./distill_ft_ds.sh
```

### Args usages
- `--from_pretrained`: Model from hugging face that nesting `AutoModelForSeq2SeqLM`
- `--dataset`: `multiclinsum`
- `--alpha`: Task weight for multi-task training.
  - $Loss = alpha * pred_l + (1 - alpha) * rationale_l$
- `--model_type`:
  - `standard`: Standard finetuning
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism

## Inference

- For distilling step-by-step models
```bash
sh ./evaluate/distill_inf.sh
```

## Datasets
* [MultiClinSum](https://zenodo.org/records/15463353)

## Known issues

* https://github.com/huggingface/evaluate/issues/609
  * `!pip install datasets==3.6.0 evaluate==0.4.3 rouge_score`
* https://github.com/huggingface/transformers/issues/36331
  * `!pip install transformers==4.45.2` from [this workaround](https://discuss.huggingface.co/t/typeerror-sentencetransformertrainer-compute-loss-got-an-unexpected-keyword-argument-num-items-in-batch/114298/4)