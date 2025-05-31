# Distil-Tuning for LM

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies.
The original project has a major limitaiton of `Seq2Seq` trainer dependencies.
The goal of the project is to bridge the gap with fine-tuning SLM LLM models (`AutoModelCasualLM`) on long-input context by heavily rely on `decoder` based models with following input [Formatting Concepts](#dataset-formatting-concepts-for-lm).

### Contribution
1. ✅ Replacement of `Seq2SeqTrainer`: `AutoModelCasualLM` models (`Qwen` series in particular).
   * Support instruction tuning
2. ✅ Refactoring and narrowing the scope, dropping dependencies.
3. ✅ Switch dependencies to `Python 3.10+`

# 🛠️ TODO
- [x] Narrow scope of the framework.
  - [x] Drop support for DeepSpeed (see [Known Issues](#known-issues))
- [x] Reforge data preparation concept (Qwen2.5 support) (see [Formatting Concepts](#dataset-formatting-concepts-for-lm))
- [x] Refactor evaluation
  - [x] Fixed `Trainer` limitation on not-exploiting `.generate` call for `predictions`
- [x] Dataset cropping
- [x] Support rationale annotation using third-party API hosting (OpenRouter)
- [x] Reforge prefix `TaskPrefixTrainer`.
  - [ ] Reforge list of parameters


## Setup

- Download `punkt_tab` for `nltk`
```bash
import nltk
nltk.download('punkt_tab')
```

## Finetuning

- Distilling step-by-step. 
```bash
./distill_ft.sh
```

### Args usages
- `--from_pretrained`: Model from hugging face that nesting `AutoModelCasualLM`
- `--dataset`: `multiclinsum`
- `--alpha`: Task weight for multi-task training.
  - $Loss = alpha * pred_l + (1 - alpha) * rationale_l$
- `--model_type`:
  - `standard`: Standard finetuning
  - `distill`: Distilling step-by-step
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
* Fine-tuning setup
  * https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune

## Known issues

* https://github.com/huggingface/transformers/blob/v4.33.2/examples/pytorch/summarization/run_summarization.py#L657
  * ROUGE calculation  
* https://github.com/huggingface/evaluate/issues/609
  * `!pip install datasets==3.6.0 evaluate==0.4.3`
* https://github.com/huggingface/transformers/issues/36331
  * `!pip install transformers==4.45.2` from [this workaround](https://discuss.huggingface.co/t/typeerror-sentencetransformertrainer-compute-loss-got-an-unexpected-keyword-argument-num-items-in-batch/114298/4)
* https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941
