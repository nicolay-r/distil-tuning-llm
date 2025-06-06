import sys
sys.path.append("..")

import argparse
import json
from os.path import join
from typing import List

import evaluate
import numpy as np
from tqdm import tqdm

from predict.cfg_multiclinsum import SUBTASKS_UNOFFICIAL, MULTICLINSUM_SUBMISSIONS
from resources.utils import load_data


def avg(obj):
    return {
        k:round(float(np.mean(v)), 4)
        for k, v in obj.items()
    }


def do_evaluate(references: List[str], predictions: List[str], args, desc=None):
    """ This is a non-official evaluator for the results.
    """

    scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore', device=args.device),
            {
                'model_type': args.bertscore_model,
                'device': args.device,
                'batch_size': 2,
                'use_fast_tokenizer': True,
            },
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
        'bluert': (
            evaluate.load('bleurt', config_name='BLEURT-20', device=args.device),
            {},
            ['scores'],
            ['bleurt']
        ),
    }

    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in tqdm(scorers.items(), desc=desc):
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    return avg(all_scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', type=int, default=1, choices=list(MULTICLINSUM_SUBMISSIONS.keys()))
    parser.add_argument('--subtask', type=str, default="test_en", choices=SUBTASKS_UNOFFICIAL)
    parser.add_argument('--bertscore_model', type=str, default="distilbert-base-uncased")
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    eval_src_filepath = join(
        "unofficial-evaluation",
        "_".join([args.subtask, str(args.run_id)]),
        "preds.json"
    )

    data = load_data(eval_src_filepath)

    all_scores_avg = do_evaluate(
        references=[item["output"] for item in data],
        predictions=[item["summary"] for item in data],
        args=args,
        desc=f"Evaluate for {eval_src_filepath}-{args.run_id}"
    )

    print(f"{eval_src_filepath}-{args.run_id}")
    print(json.dumps(all_scores_avg, indent=4))
