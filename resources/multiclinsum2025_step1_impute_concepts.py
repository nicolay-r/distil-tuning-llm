import os
from os.path import join

from cfg import DATASET_DIR
from predict.keys import OPENROUTER_API_KEY
from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content

from resources.utils import EXTRACT_PROMPT, load_data, drop_column, json_write


# Loading config.
config = load_data("multiclinsum2025_step1_config.json")
dataset_config = config["dataset_config"]
llm_config = config["llm_config"]
processing_config = config["processing_config"]

input_dataset_name = dataset_config["input_dataset_name"]
output_dataset_name = dataset_config["output_dataset_name"]

os.makedirs(join(DATASET_DIR, output_dataset_name), exist_ok=True)
src_fp_func = lambda filename: load_data(json_path=join(DATASET_DIR, input_dataset_name, filename))
tgt_fp_func = lambda filename: join(DATASET_DIR, output_dataset_name, filename)

for filename in load_data("multiclinsum2025.json")["input_files"]:

    content_it = iter_content(
        schema={"schema": [{"prompt": EXTRACT_PROMPT + ": {input}", "out": "rationale"}]},
        llm=dynamic_init(class_filepath=llm_config["provider_file"], class_name=llm_config["class_name"])(
            api_token=OPENROUTER_API_KEY,
            model_name=llm_config["model_name"]
        ),
        attempts=processing_config["attempts"],
        infer_mode=processing_config["infer_mode"],
        return_mode=processing_config["return_mode"],
        input_dicts_it=drop_column(data=src_fp_func(filename), column_name="rationale"),
    )

    json_write(
        dict_iter=content_it,
        filepath=tgt_fp_func(filename)
    )
