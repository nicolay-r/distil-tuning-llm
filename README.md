# MeDistill-d2n-long

This is the project of Medical distillation for `dialog2note` task.

## Setup
- Using Python 3.10. Setup Conda environment for finetuing:
```
conda env create -f requirements.txt
```

## Datasets
```
unzip datasets.zip
``` 

## Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`, `google/t5-v1_1-xxl`
- `--dataset`: `multiclinsum`

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

- Distilling step-by-step. 
```
./distill_ft_ds.sh
```


## Inference

- For distilling step-by-step models
```
sh ./evaluate/distill_inf.sh
```

## Reference

This work could be cited as follows:
```bibtex
@INPROCEEDINGS{xiaoxiao2024enhancing,
  author = { Liu, Xiaoxiao and Huang, Mengqing and Rusnachenko, Nicolay and Ive, Julia and Chang, Jian and Zhang, Jian Jun },
  booktitle = { 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) },
  title = {{ Enhancing Medical Dialogue Summarization: A MediExtract Distillation Framework }},
  year = {2024},
  volume = {},
  ISSN = {},
  pages = {6466-6473},
  doi = {10.1109/BIBM62325.2024.10822640},
  url = {https://doi.ieeecomputersociety.org/10.1109/BIBM62325.2024.10822640},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month =Dec
}
```

The original Distilling step-by-step project:
```bibtex
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```