import argparse
import sys

from tqdm import tqdm

sys.path.append("..")

from os.path import join, basename
from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init
from resources.utils import iter_text_files, write_text_files

from cfg import DATASET_DIR, SUMMARIZE_PROMPT_LOCALE
from keys import HF_API_KEY


submissions = {
    1: "Qwen/Qwen2.5-0.5B-Instruct",
    2: "nicolay-r/qwen25-05b-multiclinsum-standard",
    3: "nicolay-r/qwen25-05b-multiclinsum-distil"
}

subtasks = [
    "multiclinsum_test_en",
    "multiclinsum_test_es",
    "multiclinsum_test_fr",
    "multiclinsum_test_pt",
]


def run(args):

    dataset_name = args.subtask

    lang = dataset_name.split('_')[-1]

    input_dicts = list(map(
        lambda r: {"filepath": r[0], "text": r[1]},
        iter_text_files(folder_path=join(DATASET_DIR, dataset_name))
    ))

    content_it = iter_content(
        schema={"schema": [{"prompt": SUMMARIZE_PROMPT_LOCALE[lang] + ": {text}", "out": "summary"}]},
        llm=dynamic_init(class_filepath="providers/huggingface_qwen.py", class_name="Qwen2")(
            api_token=HF_API_KEY,
            model_name=submissions[args.run_id],
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

    write_text_files(
        file_iter=map(
            lambda r: (basename(r["filepath"]).split('.')[-2] + "_sum.txt", r["summary"]),
            tqdm(content_it, desc=f"{args.run_id}-{dataset_name}", total=len(input_dicts))
        ),
        folder_path=join(args.output_dir, "submissions", f"{args.team_name}_multiclinsum_{lang}_run_{args.run_id}")
    )
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--team_name', type=str, default="bu_team")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subtask', type=str, default=None, choices=subtasks)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--max_tokens', type=int, default=512)

    args = parser.parse_args()

    run(args)