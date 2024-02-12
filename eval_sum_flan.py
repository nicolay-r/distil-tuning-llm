"""
Official evaluation script for the MEDIQA-Chat 2023 shared tasks.
Adapted from: https://github.com/abachaa/MEDIQA-Chat-2023/blob/main/scripts/evaluate_summarization.py
"""
import sys
import json
import argparse

import evaluate
import pandas as pd
import numpy as np

# from sectiontagger import SectionTagger

# section_tagger = SectionTagger()


# SECTION_DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]

# # NOTE: These have been changed to match the validation set
# TASKA_RANGE = [0, 99]
# TASKA_PREFIX = "taskA"

# TASKB_RANGE = [68, 87]
# TASKB_PREFIX = "D2N"

# TASKC_RANGE = [128, 167]
# TASKC_PREFIX = "D2N"


# def add_section_divisions(row, dialogue_column):
#     row["src_len"] = len(row[dialogue_column].split())
#     for evaltype in ["reference", "prediction"]:
#         text = row[evaltype]
#         text_with_endlines = text.replace("__lf1__", "\n")
#         detected_divisions = section_tagger.divide_note_by_metasections(text_with_endlines)
#         for detected_division in detected_divisions:
#             label, _, _, start, _, end = detected_division
#             row["%s_%s" % (evaltype, label)] = text_with_endlines[start:end].replace("\n", "__lf1__")

#     return row


# def select_values_by_indices(lst, indices):
#     return [lst[ind] for ind in indices]


# def read_text(fn):
#     with open(fn, "r") as f:
#         texts = f.readlines()
#     return texts


def _validate(args, df_predictions, task_prefix, task_range):
    id_range = df_predictions.apply(lambda row: int(row[args.id_column].replace(task_prefix, "")), axis=1)
    min_id = min(id_range)
    max_id = max(id_range)
    if min_id < task_range[0] or min_id > task_range[1]:
        print("Your encounter ID range does not match the test encounters")
        sys.exit(1)
    if max_id < task_range[0] or max_id > task_range[1]:
        print("Your encounter ID range does not match the test encounters")
        sys.exit(1)
    if not args.debug and len(df_predictions) != task_range[1] - task_range[0] + 1:
        print("The number of test encounters does not match expected for this task!")
        sys.exit(1)


def test_id_range(args, df_predictions):
    # Make sure args.id_column is in range expected by task prefix (taskA or taskB)
    id_1 = df_predictions.iloc[0][args.id_column]
    if TASKA_PREFIX in id_1:
        if args.task == "taskB":
            print("Your ID prefixes do not match this tasks expected encounter_ids.")
            sys.exit(1)
        _validate(args, df_predictions, TASKA_PREFIX, TASKA_RANGE)
    elif TASKB_PREFIX in id_1:
        if args.task == "taskA":
            print("Your ID prefixes do not match this tasks expected encounter_ids.")
            sys.exit(1)
        _validate(args, df_predictions, TASKB_PREFIX, TASKB_RANGE)
    else:
        print(f"Your encounter ID -> {id_1} does not have an identifiable prefix supported by this evaluation")
        sys.exit(1)


def filter_and_aggregate(obj):
    agg_obj = {}
    for k, v in obj.items():
        # agg_obj[k] = float(np.mean([v[i] for i in indices]))
        agg_obj[k] = float(np.mean([v]))
    return agg_obj



# ==================================xiaoxiao===============================
import argparse
import os
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
import torch
import json
# from sklearn.metrics import f1_score
# from rouge_score import rouge_scorer
# from bleurt import score
# from bert_score import score as bert_score
import evaluate
from evaluate import load
import deepspeed




# 加载JSON文件
def read_test_file(test_dir):
    with open(test_dir, 'r') as file:
        data = json.load(file)
    return data

def path_process(dataset, model_type, from_pretrained, addi_info, best_step):
    model_name = from_pretrained.split("/")[1]
    if model_type == "standard":
        add_info = f"{addi_info}"
    else:
        add_info = f"_{addi_info}"
    EVAL_PATH = f"./results/{dataset}"
    eval_dir = f"{EVAL_PATH}/eval_{model_type}_{model_name}{add_info}_step{best_step}.json"
    model_dir = f"ckpts/{model_type}/{model_name}{add_info}/checkpoint-{best_step}/ckpt"
    test_dir = f'./datasets/{dataset}/{model_type}/{dataset}_test.json'
    
    # 检查result路径是否存在，不存在要创建
    if not os.path.exists(EVAL_PATH):
        os.makedirs(EVAL_PATH)
    else:
        if not os.path.exists(eval_dir):
            pass
        else:
            os.remove(eval_dir)
    
    return eval_dir, model_dir, test_dir
    

def eval(args):
    eval_dir, model_dir, test_dir = path_process(args.dataset, args.model_type, args.from_pretrained, args.addi_info, args.best_step)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained) # max_length=args.max_input_length,
    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符
    ########################33这里
    # # deepspeed_config = f'{model_dir}config.json'

    # model_engine = deepspeed.initialize(args=None,model=model,model_parameters=model.parameters(),config_params=args.deepspeed)
    # # load_path = model_dir
    # checkpoint = model_engine.load_checkpoint(model_dir, load_optimizer_states=True, load_lr_scheduler_states=True)

    ########################33上面
    checkpoint = torch.load(model_dir, map_location=device) #读取本地训练好的chekpoint
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    test_data = read_test_file(test_dir)
    # json_name = read_eval_file(args.dataset, args.model_type, args.from_pretrained, args.addi_info)
    with open(eval_dir, "a") as outfile:
        result_dict = {}
        predictions = []
        labels = []
        # 预测
        for i in range(len(test_data)):
            input_text = "predict: " + test_data[i]["input"]
            label = test_data[i]["output"]
            labels.append(label) #ground truth
        
            
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) 
            output = model.generate(input_ids) # get predicted result
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(decoded_output)
            # breakpoint()
            predictions.append(decoded_output)
            result_dict["input"] = input_text
            result_dict["prediction"] = decoded_output
            result_dict["ground_truth"] = label
            json.dump(result_dict, outfile)
    return predictions, labels


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument('--best_step', type=str, default="1000")    
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")

    args = parser.parse_args()
    
    predictions, references = eval(args)
    if len(predictions) == len(references):
        num_test = len(predictions)
    
    ######## Load Metrics from HuggingFace ########
    print("Loading ROUGE, BERTScore, BLEURT from HuggingFace")
    scorers = {
        "rouge": (
            {"path": "rouge"},
            {"use_aggregator": False},
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        ),
        "bert_scorer": (
            {"path": "bertscore"},
            {"model_type": "microsoft/deberta-xlarge-mnli", "batch_size": 1},
            ["precision", "recall", "f1"],
            ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
        ),
        "bleurt": ({"path": "bleurt", "config_name": "BLEURT-20"}, {}, ["scores"], ["bleurt"]),
    }

    ######## CALCULATE PER INSTANCE SCORES ########
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        # NOTE: We have re-written this to only load one model into memory at a time
        scores = evaluate.load(**scorer).compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]
    
    outputs = filter_and_aggregate(all_scores)
    print(outputs)
