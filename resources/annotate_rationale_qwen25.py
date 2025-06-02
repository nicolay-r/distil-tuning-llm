import os
from os.path import join

from keys import API_KEY
from utils import load_data, json_write, DATASETS_DIR, EXTRACT_PROMPT, drop_column
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content

input_dataset_name = "multiclinsum"
output_dataset_name = "multiclinsum_rationale"
input_files = [
    "multiclinsum_gs_en.json",
    "multiclinsum_gs_es.json",
    "multiclinsum_gs_fr.json",
    "multiclinsum_gs_pt.json"
]

os.makedirs(join(DATASETS_DIR, output_dataset_name), exist_ok=True)

for filename in input_files:

    content_it = iter_content(
        schema={
            "schema": [
                {"prompt": EXTRACT_PROMPT + ": {input}", "out": "rationale"}
            ]
        },
        llm=dynamic_init(class_filepath="open_router.py", class_name="OpenRouter")(
            api_token=API_KEY,
            model_name="qwen/qwen2.5-72b-instruct"
        ),
        attempts=100,
        infer_mode="single",
        return_mode="record",
        input_dicts_it=drop_column(
            data=load_data(json_path=join(DATASETS_DIR, input_dataset_name, filename)),
            column_name="rationale"
        ),
    )

    json_write(
        dict_iter=content_it,
        filepath=join(DATASETS_DIR, output_dataset_name, filename)
    )
