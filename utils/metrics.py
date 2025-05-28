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
from rouge_score import rouge
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download("punkt", quiet=True)

rouge = evaluate.load("rouge")


def compute_metrics_equation(tokenizer):

    def __sanitize_text(text: str, lowercase: bool = False) -> str:
        """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
        sanitized_text = " ".join(text.strip().split())
        sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
        return sanitized_text

    def __postprocess_text(preds, labels):
        preds = [__sanitize_text(pred) for pred in preds]
        labels = [__sanitize_text(label) for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels[0], skip_special_tokens=True)
        decoded_preds, decoded_labels = __postprocess_text(decoded_preds, decoded_labels)

        result = {}

        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result.update(rouge_results)

        result["rouge_avg"] = np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]).item()

        result = {k: round(v * 100, 4) for k, v in result.items()}

        # Add length of generated and reference summaries
        generated_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        reference_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
        result["mean_generated_len"] = np.mean(generated_lens)
        result["mean_reference_len"] = np.mean(reference_lens)
        wandb.log(result)
        return result

    return compute_metrics
