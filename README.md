# Distil-Tuning for LM
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TXGaz39o73nBucEQw12gbad7Tw11j2Ol?usp=sharing)

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies.
The original project has a major limitation of `Seq2Seq` trainer dependencies.
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
  - [x] Reforge list of parameters


## Setup

```bash
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


## References

* bulk-chain: https://github.com/nicolay-r/bulk-chain
  * Annotation and test-set inference. 