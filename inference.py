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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载JSON文件
# with open('./datasets/svamp/svamp_valid.json', 'r') as file:
with open('./datasets/svamp/task_prefix/svamp_test.json', 'r') as file:
    data = json.load(file)
# labels = [item['label'] for item in data]
# # 提取预测和标签
# input_texts = [item['input'] for item in data]


# 准备模型
model_name = "t5-v1_1-small"
task_type = 'task_prefix'

from_pretrained = "google/{}".format(model_name)

tokenizer = AutoTokenizer.from_pretrained(from_pretrained)

model = T5ForConditionalGeneration.from_pretrained(from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符

# model_path = "./ckpts/svamp/{}/{}/gt/1.0/gt/0.5/1024/4/AdamW/5e-05/0/checkpoint-10000/pytorch_model.bin".format(model_name, task_type)
# model_path = "./ckpts/svamp/t5-v1_1-base/standard/gt/1.0/gt/0.5/1024/2/AdamW/5e-05/0/checkpoint-10000/pytorch_model.bin"
model_path = "./ckpts/task_prefix/t5-v1_1-small-before/0/checkpoint-10000/pytorch_model.bin"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)

model.eval()

# input_text = "Hi hello"

json_name = "./results/svamp/evaluation_{}_{}_2.json".format(model_name, task_type)
with open(json_name, "a") as outfile:
    result_dict = {}
    predictions = []
    labels = []
    # 预测
    for i in range(len(data)):
        input_text = "predict: " + data[i]["input"]
        label = data[i]["output"]
        labels.append(label)
    # for input_text in input_texts:
        
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) 
        output = model.generate(input_ids)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(decoded_output)
        breakpoint()
        predictions.append(decoded_output)
        result_dict["input"] = input_text
        result_dict["prediction"] = decoded_output
        result_dict["ground_truth"] = label
        json.dump(result_dict, outfile)

# 计算F1分数
# 注意：F1分数通常用于分类问题。这里我们假设'predictions'和'labels'是二分类标签。
# f1 = f1_score(labels, predictions, average = None)

# # 计算ROUGE-1
# scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
# rouge1_scores = [scorer.score(label, prediction)['rouge1'].fmeasure for label, prediction in zip(labels, predictions)]
# avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
rouge = evaluate.load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
rouge1_scores = rouge.compute(predictions=predictions,references=labels)
# breakpoint()
# avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)

# 计算BLEURT
# bleurt_scorer = score.BleurtScorer('bleurt_checkpoint')
# bleurt_scores = [bleurt_scorer.score(references=[label], candidates=[prediction])[0] for label, prediction in zip(labels, predictions)]
# avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

bleurt = load("bleurt", module_type="metric")
bleurt_scores = bleurt.compute(predictions=predictions, references=labels)['scores']
# breakpoint()
avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

# # 计算BERTScore
# P, R, F1 = bert_score(labels, predictions, lang='en', verbose=True)
# avg_bertscore = F1.mean().item()

bertscore = load("bertscore")
bertscores = bertscore.compute(predictions=predictions, references=labels, lang="en")['precision']
avg_bertscore = sum(bertscores) / len(bertscores)



# print(f'F1 Score: {f1}')
print(f'Average ROUGE-1: {rouge1_scores}')
print(f'Average BLEURT: {avg_bleurt}')
print(f'Average BERTScore: {avg_bertscore}')

