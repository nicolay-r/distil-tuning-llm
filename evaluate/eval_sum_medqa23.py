import sys

sys.path.append("..")  # 添加上层目录到路径中，使得 utils 模块可以被找到
import json
import argparse
import torch
import evaluate
import pandas as pd
import numpy as np

from utils.sectiontagger import SectionTagger
section_tagger = SectionTagger()


# SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

TASKA_RANGE = [0,199]
TASKA_PREFIX = ''

TASKB_RANGE = [88,127]
TASKB_PREFIX = 'D2N'

TASKC_RANGE = [128,167]
TASKC_PREFIX = 'D2N'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# def add_section_divisions(row, dialogue_column ):
#     row['src_len'] = len(row[ dialogue_column ].split())
#     for evaltype in ['output', 'prediction']:
#         text = row[evaltype]
#         text_with_endlines = text.replace( '__lf1__', '\n' )
#         detected_divisions = section_tagger.divide_note_by_metasections(text_with_endlines)
#         for detected_division in detected_divisions:
#             label, _, _, start, _, end = detected_division
#             row[ '%s_%s' % (evaltype, label)] = text_with_endlines[start:end].replace('\n', '__lf1__')

#     return row


# def select_values_by_indices(lst, indices) :
#     return [lst[ind] for ind in indices]


# def read_text(fn):
#     with open(fn, 'r') as f:
#         texts = f.readlines()
#     return texts


# def _validate(args, df_predictions, task_prefix, task_range):
#     id_range = df_predictions.apply(lambda row: int( str(row[args.id_column]).replace(task_prefix, '')), axis=1)
#     min_id = min(id_range)
#     max_id = max(id_range)
#     if min_id < task_range[0] or min_id > task_range[1]:
#         print('Your encounter ID range does not match the test encounters')
#         sys.exit(1)
#     if max_id < task_range[0] or max_id > task_range[1]:
#         print('Your encounter ID range does not match the test encounters')
#         sys.exit(1)
#     if not args.debug and len(df_predictions) != task_range[1] - task_range[0] + 1:
#         print('The number of test encounters does not match expected for this task!')
#         sys.exit(1)



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

    parser.add_argument(
        '--task', action='store', default='taskA',
        help='summarization task, default is for full note (taskB). (use snippet, taskA, otherwise).'
    )
    
    args = parser.parse_args()

    # Only need id and prediction from df_predictions
    full_df = pd.read_csv(f"{args.fn_eval_data}/generated_predictions_df.csv") 
    # full_df = pd.read_csv(f"{args.fn_eval_data}/updated_predictions_df.csv") 
    
    # 
    # full_df = df_references.merge(df_predictions[[args.id_column, 'prediction']], on=args.id_column)
    full_df['dataset'] = 0

    # create lists for references/predictions so we only need to calculate the scores once per instance
    references = full_df['output'].tolist()
    predictions = full_df['prediction'].tolist()
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
       
