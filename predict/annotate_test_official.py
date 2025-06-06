import sys
sys.path.append("..")

import argparse
from tqdm import tqdm
from os.path import join, basename
from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init
from resources.utils import iter_text_files, write_text_files

from cfg import DATASET_DIR, SUMMARIZE_PROMPT_LOCALE
from cfg_multiclinsum import MULTICLINSUM_SUBMISSIONS, SUBTASKS_OFFICIAL
from keys import HF_API_KEY


def fmt_filepath_summary(filepath):
    return basename(filepath).split('.')[-2] + "_sum.txt"


def run(args, input_dicts, lang):
    return iter_content(
        schema={"schema": [{"prompt": SUMMARIZE_PROMPT_LOCALE[lang] + ": {input}", "out": "summary"}]},
        llm=dynamic_init(class_filepath="providers/huggingface_qwen.py", class_name="Qwen2")(
            api_token=HF_API_KEY,
            model_name=MULTICLINSUM_SUBMISSIONS[args.run_id],
            temp=0.1,
            use_bf16=True,
            max_new_tokens=args.max_tokens,
            device=args.device
        ),
        infer_mode="batch",
        batch_size=args.batch_size,
        return_mode="record",
        input_dicts_it=input_dicts,
    )

        
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

    content_it = run(args, input_dicts=input_dicts, target_dir=target_dir, lang=lang)

    write_text_files(
        file_iter=map(
            lambda r: (fmt_filepath_summary(r["filepath"]), r["summary"]),
            tqdm(content_it, desc=f"{args.run_id}-{args.subtask}", total=len(input_dicts))
        ),
        folder_path=target_dir
    )
