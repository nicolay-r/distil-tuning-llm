import numpy as np
import evaluate
import nltk
import wandb
from transformers import EvalPrediction, AutoTokenizer


class RougeSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = evaluate.load("rouge")
        return cls._instance


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


def compute_metrics_rouge(eval_preds: EvalPrediction, tokenizer: AutoTokenizer):

    # IMPORTANT:
    # The original Trainer instance does not perform .generate() call, which is important for MLM
    # Therefore, we expected to have `logits` as predictions from CLMs.
    # Those logits should be converted into tokens by selecting the likely one.
    preds = np.argmax(eval_preds.predictions, axis=-1)
    labels = eval_preds.label_ids

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = {}

    metric = RougeSingleton.get_instance()

    rouge_results = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result.update(rouge_results)

    result["rouge_avg"] = np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]).item()
    print(result)

    result = {k: round(v * 100, 4) for k, v in result.items()}

    # Add length of generated and reference summaries
    generated_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    reference_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
    result["mean_generated_len"] = np.mean(generated_lens)
    result["mean_reference_len"] = np.mean(reference_lens)

    wandb.log(result)

    return result
