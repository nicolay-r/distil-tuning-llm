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
import nltk
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download("punkt", quiet=True)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)


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
    rouge = evaluate.load(
        "rouge",
        download_config=datasets.DownloadConfig(cache_dir=model_args.cache_dir, local_files_only=True, use_etag=False)
        if is_offline_mode()
        else None,
        # cache_dir=model_args.cache_dir,
    )
    bertscore = evaluate.load(
        "bertscore",
        download_config=datasets.DownloadConfig(cache_dir=model_args.cache_dir, local_files_only=True, use_etag=False)
        if is_offline_mode()
        else None,
        cache_dir="./",
    )
    bleurt = evaluate.load(
        "bleurt",
        "BLEURT-20",
        # Don't ask me why, but BLEURT needs a different download_config than the other metrics
        download_config=datasets.DownloadConfig(use_etag=False) if is_offline_mode() else None,
        # cache_dir=model_args.cache_dir,
    )
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
    
    def compute_metrics(eval_pred):
        # predictions, labels = eval_pred
        # # breakpoint()
        # ps = np.where(predictions[0] != -100,predictions[0],tokenizer.pad_token_id)
        # decoded_preds = tokenizer.batch_decode(ps, skip_special_tokens=True)
        
        # ls = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # # preds = np.where(preds != -100,preds,tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(ls, skip_special_tokens=True)

        # scorers = {
        #     'rouge': (
        #         evaluate.load('rouge'),
        #         {'use_aggregator': False},
        #         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        #         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        #     ),
        # }
        # all_scores = {}
        # for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        #     scores = scorer.compute(references=decoded_labels, predictions=decoded_preds, **kwargs)
        #     for score_key, save_key in zip(keys, save_keys):
        #         all_scores[save_key] = scores[score_key]
        # breakpoint()
        # acc = np.mean(all_scores['rougeL'])
        # wandb.log({'eval/acc': acc,              
        #         },
        #         step=self.state.global_step)
        # return {'accuracy': acc}
        
        # breakpoint()
        predictions, labels = eval_pred
        # breakpoint()
        ps = np.where(predictions[0] != -100,predictions[0],tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(ps, skip_special_tokens=True)
        
        ls = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # preds = np.where(preds != -100,preds,tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(ls, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}
        
        # For TaskA, we need to extract the header and text from the predictions and labels
        # if data_args.task == TASK_A:
        #     decoded_preds, header_preds = extract_header_and_text(decoded_preds)
        #     decoded_labels, header_labels = extract_header_and_text(decoded_labels)

            # # Compute section header metrics
            # result.update(exact_match.compute(predictions=header_preds, references=header_labels))

        # Compute section text metrics...

        # ROUGE
        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result.update(rouge_results)

        # Compute the arithmetic mean of ROUGE-1, ROUGE-2 and ROUGE-L following: https://arxiv.org/abs/2110.08499
        result["rouge_avg"] = np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]).item()

        # # BERTScorewatc
        # bertscore_result = bertscore.compute(
        #     predictions=decoded_preds,
        #     references=decoded_labels,
        #     batch_size=24,
        #     device=device,
        #     # These are mostly based on the recommendations in https://github.com/Tiiiger/bert_score
        #     model_type="microsoft/deberta-xlarge-mnli",
        #     lang="en",
        #     rescale_with_baseline=True,
        #     use_fast_tokenizer=True,
        # )
        # result.update(
        #     {
        #         "bertscore_p": np.mean(bertscore_result["precision"]).item(),
        #         "bertscore_r": np.mean(bertscore_result["recall"]).item(),
        #         "bertscore_f1": np.mean(bertscore_result["f1"]).item(),
        #     }
        # )

        # # BLEURT
        # bleurt_result = bleurt.compute(predictions=decoded_preds, references=decoded_labels)
        # result.update({"bleurt": np.mean(bleurt_result["scores"]).item()})
        # breakpoint()
        # Compute an ensemble score for the generations
        # result["ensemble_gen_score"] = np.mean(result["rouge_avg"]).item()
        result = {k: round(v * 100, 4) for k, v in result.items()}

        # Add length of generated and reference summaries
        generated_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        reference_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
        result["mean_generated_len"] = np.mean(generated_lens)
        result["mean_reference_len"] = np.mean(reference_lens)
        
        return result
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