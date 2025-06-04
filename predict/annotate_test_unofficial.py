import sys
sys.path.append("..")

import argparse
from os.path import join

from cfg import DATASET_DIR
from predict.annotate_test_official import run
from resources.utils import json_write, load_data

subtasks = [
    "test_en",
    "test_es",
    "test_fr",
    "test_pt",
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--team_name', type=str, default="bu_team")
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subtask', type=str, default=None, choices=subtasks)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--max_tokens', type=int, default=512)

    args = parser.parse_args()

    lang = args.subtask.split('_')[-1]
    target_dir = join(args.output_dir, "submissions", f"{args.subtask}_{args.run_id}")
    input_dicts = load_data(json_path=join(DATASET_DIR, "multiclinsum_rationale_mult", f"{args.subtask}.json"))

    content_it = run(args, input_dicts=input_dicts, lang=lang)

    json_write(dict_iter=content_it, filepath=join(target_dir, 'preds.json'))
