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
import pandas as pd




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
    model_dir = f"./ckpts/{model_type}/{model_name}{add_info}/checkpoint-{best_step}/pytorch_model.bin"
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
    
    SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
    full_df = pd.DataFrame(columns = SECTION_DIVISIONS)
    eval_dir, model_dir, test_dir = path_process(args.dataset, args.model_type, args.from_pretrained, args.addi_info, args.best_step)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained) # max_length=args.max_input_length,
    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符
    checkpoint = torch.load(model_dir, map_location="cpu") #读取本地训练好的chekpoint
    model.load_state_dict(checkpoint)
    # 如果有多个 GPU，则使用 DataParallel 来利用它们
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
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
            output = model.module.generate(input_ids) # get predicted result
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(decoded_output)
            # breakpoint()
            predictions.append(decoded_output)
            result_dict["input"] = input_text
            result_dict["prediction"] = decoded_output
            result_dict["ground_truth"] = label
            json.dump(result_dict, outfile)
        full_df['reference'] = labels
        full_df['prediction'] = predictions
    print(full_df.head(5))
    print(full_df.columns)
    csv_path = './full_df.csv'
    full_df.to_csv(csv_path)

    # # 计算ROUGE-1

    rouge = evaluate.load('rouge')
    
    rouge1_scores = rouge.compute(predictions=predictions,references=labels)
    
    # 计算BLEURT
    bleurt = load("bleurt", module_type="metric")
    bleurt_scores = bleurt.compute(predictions=predictions, references=labels)['scores']
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

    # # 计算BERTScore
    bertscore = load("bertscore")
    bertscores = bertscore.compute(predictions=predictions, references=labels, lang="en")['precision']
    avg_bertscore = sum(bertscores) / len(bertscores)

    print(f'Results for task {args.model_type} on model {args.from_pretrained} with{args.addi_info}')
    print(f'Average ROUGE-1: {rouge1_scores}')
    print(f'Average BLEURT: {avg_bleurt}')
    print(f'Average BERTScore: {avg_bertscore}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # args.dataset, args.model_type, args.from_pretrained, args.addi_info
    
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument('--best_step', type=str, default="1000")
    args = parser.parse_args()

    
    # data = read_test_file(args.path_file)
    eval(args)