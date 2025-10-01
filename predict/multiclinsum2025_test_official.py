import sys
sys.path.append("..")

import argparse
from tqdm import tqdm
from os.path import join, basename
from resources.utils import iter_text_files, write_text_files
from utils import infer_summary

from cfg import DATASET_DIR
from cfg_multiclinsum import MULTICLINSUM_SUBMISSIONS, SUBTASKS_OFFICIAL


def fmt_filepath_summary(filepath):
    return basename(filepath).split('.')[-2] + "_sum.txt"

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--team_name', type=str, default="bu_team")
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subtask', type=str, default=None, choices=SUBTASKS_OFFICIAL)
    parser.add_argument('--run_id', type=int, default=None, choices=list(MULTICLINSUM_SUBMISSIONS.keys()))
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--max_tokens', type=int, default=512)

    args = parser.parse_args()

    lang = args.subtask.split('_')[-1]
    target_dir = join(args.output_dir, "submissions", f"{args.team_name}_multiclinsum_{lang}_run_{args.run_id}")
    input_dicts = list(map(
        lambda r: {"filepath": r[0], "input": r[1]},
        iter_text_files(folder_path=join(DATASET_DIR, args.subtask),
                        skip_if_exists_in=target_dir,
                        max_content_length=args.max_input_length,
                        fmt_filename_func=lambda filepath: fmt_filepath_summary(filepath))
    ))

    content_it = infer_summary(args, input_dicts=input_dicts, target_dir=target_dir, lang=lang)

    write_text_files(
        file_iter=map(
            lambda r: (fmt_filepath_summary(r["filepath"]), r["summary"]),
            tqdm(content_it, desc=f"{args.run_id}-{args.subtask}", total=len(input_dicts))
        ),
        folder_path=target_dir
    )
