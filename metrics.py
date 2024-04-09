# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import evaluate
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        prediction_0 = predictions[0]
        prediction_0 = np.where(prediction_0 != -100, prediction_0, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(prediction_0, skip_special_tokens=True)
        # 使用 np.where 来处理标签数据。当标签不等于 -100 时，保留原标签；否则，替换为 pad_token_id。
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics



def compute_metrics_equation(tokenizer):
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # breakpoint()
        ps = np.where(predictions[0] != -100,predictions[0],tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(ps, skip_special_tokens=True)
        
        ls = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # preds = np.where(preds != -100,preds,tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(ls, skip_special_tokens=True)

        scorers = {
            'rouge': (
                evaluate.load('rouge'),
                {'use_aggregator': False},
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            ),
        }
        all_scores = {}
        for name, (scorer, kwargs, keys, save_keys) in scorers.items():
            scores = scorer.compute(references=decoded_labels, predictions=decoded_preds, **kwargs)
            for score_key, save_key in zip(keys, save_keys):
                all_scores[save_key] = scores[score_key]
        # breakpoint()
        acc = np.mean(all_scores['rouge1'])
        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):

    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # breakpoint()
        # preds = np.where(predictions[0] != -100,predictions[0],tokenizer.pad_token_id)
        ps = np.where(predictions != -100,predictions,tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(ps, skip_special_tokens=True)
        
        ls = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # preds = np.where(preds != -100,preds,tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(ls, skip_special_tokens=True)
        # breakpoint()
        # preds = list()
        # for pred in decoded_preds:    
        #     preds.append(eval_equation(pred))

        # lbs = list()
        # for label in decoded_labels:    
        #     lbs.append(eval_equation(label))
        # breakpoint()
        # acc = np.mean(np.array(preds) == np.array(labels))
        scorers = {
            'rouge': (
                evaluate.load('rouge'),
                {'use_aggregator': False},
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            ),
            # 'bert_scorer': (
            #     evaluate.load('bertscore', device=device),
            #     {'model_type': 'microsoft/deberta-xlarge-mnli', 'device':device},
            #     ['precision', 'recall', 'f1'],
            #     ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
            # ),
            # 'bluert': (
            #     evaluate.load('bleurt', config_name='BLEURT-20', device=device),
            #     {},
            #     ['scores'],
            #     ['bleurt']
            # ),
        }
        all_scores = {}
        for name, (scorer, kwargs, keys, save_keys) in scorers.items():
            scores = scorer.compute(references=decoded_labels, predictions=decoded_preds, **kwargs)
            for score_key, save_key in zip(keys, save_keys):
                all_scores[save_key] = scores[score_key]
        # breakpoint()
        acc = np.mean(all_scores['rouge1'])
        return {'accuracy': acc}
    
    return compute_metrics
    
    return compute_metrics