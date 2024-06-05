import sys
import nltk
sys.path.append("..")  # 添加上层目录到路径中，使得 utils 模块可以被找到
import json
import argparse
import torch
import evaluate
import pandas as pd
import numpy as np

from utils.sectiontagger import SectionTagger
section_tagger = SectionTagger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text
def postprocess_text(preds, labels):

    preds = [sanitize_text(pred) for pred in preds]
    labels = [sanitize_text(label) for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def filter_and_aggregate(obj, indices):
    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, to_email):
    from_email = 'rosaliu.567@gmail.com'
    with open("../app_keys/k.txt", 'r') as k:
        psw = k.read()
    
    password = psw
    k.close()
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    body = MIMEText(message, 'plain')
    msg.attach(body)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
        prog='evaluate_summarization',
        description='This runs basic evaluation for both snippet (taskA) and full note summarization (taskB).'
    )

    parser.add_argument('--fn_eval_data', required=True, help='filename of gold references requires id and note column.')

    # parser.add_argument(
    #     '--task', action='store', default='taskA',
    #     help='summarization task, default is for full note (taskB). (use snippet, taskA, otherwise).'
    # )
    
    args = parser.parse_args()
    
    csv_path = f"{args.fn_eval_data}/generated_predictions_df.csv"
    # Only need id and prediction from df_predictions
    full_df = pd.read_csv(csv_path)
    # full_df = df_references.merge(df_predictions[[args.id_column, 'prediction']], on=args.id_column)
    full_df['dataset'] = 0

    # create lists for references/predictions so we only need to calculate the scores once per instance
    references = full_df['output'].tolist()
    predictions = full_df['prediction'].tolist()
    
    # predictions, references  = postprocess_text(full_df['output'].tolist(), full_df['prediction'].tolist())
    
    num_test = len(full_df)

    ######## Load Metrics from HuggingFace ########
    print('Loading ROUGE, BERTScore, BLEURT from HuggingFace')
    scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore', device=device),
            {'model_type': 'microsoft/deberta-xlarge-mnli', 'device':device},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
        'bluert': (
            evaluate.load('bleurt', config_name='BLEURT-20', device=device),
            {},
            ['scores'],
            ['bleurt']
        ),
    }

    ######## CALCULATE PER INSTANCE SCORES ########
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    cohorts = [
        ('all', list(range(num_test))),
    ]

    subsets = full_df['dataset'].unique().tolist()
    for subset in subsets:
        # Don't include anything after num_test (section-level)
        indices = full_df[full_df['dataset'] == subset].index.tolist()
        cohorts.append((f'dataset-{subset}', indices))



    outputs = {k: filter_and_aggregate(all_scores, idxs) for (k, idxs) in cohorts}

    # ###### OUTPUT TO JSON FILE ########

    for cohort, obj in outputs.items():
        print(cohort)
        for k, v in obj.items():
            print(f'\t{k} -> {round(v, 3)}')
        print('\n')
    
    
    
    to_email = "rosaliu.567@gmail.com"
    # to_email = "rosaliu.567@gmail.com"
    send_email('分数出炉啦！', f'您的模型评估已完成。model:{args.fn_eval_data}: {outputs}', to_email)
       
