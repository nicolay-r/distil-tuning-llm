import os
from os.path import join

from cfg import DATASET_DIR
from predict.keys import OPENROUTER_API_KEY
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content

from resources.utils import EXTRACT_PROMPT, load_data, drop_column, json_write

input_dataset_name = "multiclinsum"
output_dataset_name = "multiclinsum_rationale"
input_files = [
    "multiclinsum_gs_en.json",
    "multiclinsum_gs_es.json",
    "multiclinsum_gs_fr.json",
    "multiclinsum_gs_pt.json"
]

os.makedirs(join(DATASET_DIR, output_dataset_name), exist_ok=True)
src_fp_func = lambda filename: load_data(json_path=join(DATASET_DIR, input_dataset_name, filename))
tgt_fp_func = lambda filename: load_data(json_path=join(DATASET_DIR, output_dataset_name, filename))

for filename in input_files:

    content_it = iter_content(
        schema={"schema": [{"prompt": EXTRACT_PROMPT + ": {input}", "out": "rationale"}]},
        llm=dynamic_init(class_filepath="providers/open_router.py", class_name="OpenRouter")(
            api_token=OPENROUTER_API_KEY,
            model_name="qwen/qwen2.5-72b-instruct"
        ),
        attempts=100,
        infer_mode="single",
        return_mode="record",
        input_dicts_it=drop_column(data=src_fp_func(filename), column_name="rationale"),
    )

    json_write(
        dict_iter=content_it,
        filepath=tgt_fp_func(filename)
    )
