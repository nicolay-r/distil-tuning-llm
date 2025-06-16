# Distil-Tuning for Decoder-Based Transformers
![](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TXGaz39o73nBucEQw12gbad7Tw11j2Ol?usp=sharing)

This repo represent a tiny and **reforged version** of the original [`MeDistil-d2n` framework](https://github.com/Xiaoxiao-Liu/distill-d2n) and the related paper studies for the [BioASQ workshop](https://bioasq.org/) on multilingual clinical texts summarization.
The original project has a major limitation of `Seq2Seq` trainer dependencies.
The goal of the project is to bridge the gap with fine-tuning SLM LLM models (`AutoModelCasualLM`) on long-input context by heavily rely on `decoder` based models with following input [Formatting Concepts](#dataset-formatting-concepts-for-lm).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64e62d11d27a8292c3637f86/cphY-l_YKa4JehUM3fIUT.png)

### Contribution
1. ✅ Replacement of `Seq2SeqTrainer`: `AutoModelCasualLM` models (`Qwen` series in particular).
   * Support instruction tuning
2. ✅ Refactoring and narrowing the scope, dropping dependencies.
3. ✅ Switch dependencies to `Python 3.10+`

### 🛠️ Changeset
- [x] Narrow scope of the framework. We don not support DeepSpeed by default
- [x] Reforge data preparation concept (Qwen2.5 support) (see [Formatting Concepts](#input-formatting-concepts))
- [x] Refactor evaluation
  - [x] Fixed `Trainer` limitation on not-exploiting `.generate` call for `predictions`
- [x] Dataset cropping
- [x] Support rationale annotation using third-party API hosting (OpenRouter)
- [x] Reforge prefix `TaskPrefixTrainer`.
  - [x] Reforge list of parameters
- [ ] ‼️**Memory leakage on evaluation**
  - Caused by this piece: https://github.com/nicolay-r/distill-tuning-llm/blob/07871555069ef07a8149e51b36ba6381dad4b423/utils/distill_trainer.py#L84 


# Setup

* The complete list of dependencies

```bash
pip install -r requirements.txt
```

- Download `punkt_tab` for `nltk`
```bash
import nltk
nltk.download('punkt_tab')
```

# Finetuning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TXGaz39o73nBucEQw12gbad7Tw11j2Ol?usp=sharing)

Manual Training:

```bash
./distill_ft_qwen25_test.sh --from_pretrained "AutoModelCasualLM-from-HF" --dataset "multiclinsum" --model_type "distill"
```

> **NOTE**: We use the following [post-processing](https://github.com/nicolay-r/distill-tuning-llm/blob/main/resources/make_dataset_mult.py) script for dataset preparation. 

List of the parameters
- `--from_pretrained`: Model from hugging face that nesting `AutoModelCasualLM`
- `--dataset`: `multiclinsum` (see [downloading script](https://github.com/nicolay-r/distill-tuning-llm/blob/main/resources/download_dataset.sh) and [post-processing](https://github.com/nicolay-r/distill-tuning-llm/blob/main/resources/make_dataset_mult.py))
- `--alpha`: Task weight for multi-task training.
  - $Loss = alpha * pred_l + (1 - alpha) * rationale_l$
- `--model_type`:
  - `standard`: Standard finetuning (baseline)
  - `distill`: Distilling step-by-step

The pretrained models are publicly available:
| Model 🤗         | Link                                               |
|------------------|----------------------------------------------------|
| `nicolay-r/qwen25-05b-multiclinsum-distil`       | [model-card](https://huggingface.co/nicolay-r/qwen25-05b-multiclinsum-distil)       |
| `nicolay-r/qwen25-05b-multiclinsum-standard`       | [model-card](https://huggingface.co/nicolay-r/qwen25-05b-multiclinsum-standard)   |

# Inference

We use [`bulk-chain` project](https://github.com/nicolay-r/bulk-chain) to infer:
* `rationale` prompts, necessary for distill-based fine-tuning [[using this script].](https://github.com/nicolay-r/distill-tuning-llm/blob/main/predict/annotate_train_rationale.py)
* Test data for competition submissions [[using this script]](https://github.com/nicolay-r/distill-tuning-llm/blob/main/predict/annotate_test_official.py)

# Datasets
* **MultiClinSum**
  * We use the [following script](https://github.com/nicolay-r/distill-tuning-llm/blob/main/resources/download_dataset.sh) for downloading datasets.
  * **Web**: https://temu.bsc.es/multiclinsum 
  * **Data**: https://zenodo.org/records/15463353
  * **BioASQ**: http://bioasq.org/ 
   
## Input formatting concepts

* Data formatting for QWEN
  * https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#control-tokens-chat-template
* Fine-tuning setup
  * https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune

## References

* bulk-chain: https://github.com/nicolay-r/bulk-chain
  * Annotation and test-set inference. 
