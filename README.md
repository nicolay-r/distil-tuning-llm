# Distil-Tuning for Text Summarization with `AutoModelCasualLM`

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies.
The original project has a major limitaiton of `Seq2Seq` trainer dependencies.
The goal of the project is to bridge the gap with fine-tuning SLM LLM models on long-input context by heavily rely on `decoder` based models with following input [Formatting Concepts](#dataset-formatting-concepts-for-lm).

### Contribution
1. ✅ Replacement of `Seq2SeqTrainer`: `AutoModelCasualLM` models (`Qwen` series in particular).
   * Support instruction tuning
2. ✅ Refactoring and narrowind the scope, droppping dependencies.
3. ✅ Switch dependencies to `Python 3.10+`

# 🛠️ TODO
- [x] Narrow scope of the framework.
  - [x] Drop support for DeepSpeed (see [Known Issues](#known-issues))
- [x] Reforge data preparation concept (Qwen2.5 support) (see [Formatting Concepts](#dataset-formatting-concepts-for-lm))
- [ ] Reforge evaluation.
- [ ] Reforge prefix `TaskPrefixTrainer`.
  - [ ] Sync parameters list with one at data preparation stage
  - [ ] Reforge list of parameters


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
- `--from_pretrained`: Model from hugging face that nesting `AutoModelCasualLM`
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

## Dataset Formatting Concepts for LM

* Data formatting for QWEN
  * https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#control-tokens-chat-template 

## Known issues

* https://github.com/huggingface/evaluate/issues/609
  * `!pip install datasets==3.6.0 evaluate==0.4.3 rouge_score`
* https://github.com/huggingface/transformers/issues/36331
  * `!pip install transformers==4.45.2` from [this workaround](https://discuss.huggingface.co/t/typeerror-sentencetransformertrainer-compute-loss-got-an-unexpected-keyword-argument-num-items-in-batch/114298/4)
* https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941
