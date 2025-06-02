import sys

from tqdm import tqdm

sys.path.append("..")

from os.path import join, basename
from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init
from resources.utils import iter_text_files, write_text_files

from cfg import DATASET_DIR, SUMMARIZE_PROMPT_LOCALE
from keys import HF_API_KEY


team_name = "bu_team"

submissions = {
    1: "Qwen/Qwen2.5-0.5B-Instruct",
    2: "nicolay-r/qwen25-05b-multiclinsum-standard",
    3: "nicolay-r/qwen25-05b-multiclinsum-distil"
}

subtasts = [
   "multiclinsum_test_en",
   "multiclinsum_test_es",
   "multiclinsum_test_fr",
   "multiclinsum_test_pt",
]

for i in range(len(submissions)):

    run_id = i + 1
    model_name = submissions[run_id]

    for dataset_name in subtasts:

        lang = dataset_name.split('_')[-1]

        input_dicts = list(map(
            lambda r: {"filepath": r[0], "text": r[1]},
            iter_text_files(folder_path=join(DATASET_DIR, dataset_name))
        ))

        content_it = iter_content(
            schema={"schema": [{"prompt": SUMMARIZE_PROMPT_LOCALE[lang] + ": {text}", "out": "summary"}]},
            llm=dynamic_init(class_filepath="providers/huggingface_qwen.py", class_name="Qwen2")(
                api_token=HF_API_KEY,
                model_name=model_name,
                temp=0.1,
                max_new_tokens=1024,
                device='cuda'
            ),
            infer_mode="batch",
            batch_size=10,
            return_mode="record",
            input_dicts_it=input_dicts,
        )

        write_text_files(
            file_iter=map(
                lambda r: (basename(r["filepath"]).split('.')[-2] + "_sum.txt", r["summary"]),
                tqdm(content_it, desc=dataset_name, total=len(input_dicts))
            ),
            folder_path=join("submissions", f"{team_name}_multiclinsum_{lang}_run_{run_id}")
        )