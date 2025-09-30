import sys
sys.path.append("..")

import os
from cfg import DATASET_DIR
from os.path import join
from utils import split_dataset, json_save_list, drop_column, load_data


# Load configuration
config = load_data(json_path="cfg_multiclinsum2025.json")

data_cfg = config["dataset_config"]

# Extract parameters from config
output_dataset_name = data_cfg["output_dataset_name"]
input_max_length = data_cfg["input_max_length"]
output_max_length = data_cfg["output_max_length"]

train_data = []
valid_data = []
test_data = {}      # We do not merge test data.
data_by_type = {
    "train": train_data,
    "valid": valid_data
}

for filename in data_cfg["input_files"]:

    train, valid, test = split_dataset(
        json_path=join(DATASET_DIR, data_cfg["input_dataset_name"], filename),
        train_ratio=data_cfg["train_ratio"],
        valid_ratio=data_cfg["valid_ratio"],
        test_ratio=data_cfg["test_ratio"]
    )

    train_data += train
    valid_data += valid

    # For the test data we consider separated statistic.
    test_data[filename] = test

# Remove non utilized strings.
drop_column(train_data, column_name="rationale_prompt")
drop_column(valid_data, column_name="rationale_prompt")
# For test data we perform this operation individually for each subtask.
for data in test_data.values():
    drop_column(data, column_name="rationale_prompt")
    drop_column(data, column_name="rationale")             # For the test data we're also dropping `rationale`.

# Crop data based on config.
crop_cfg = config["processing_config"]["crop_data"]
if crop_cfg["enabled"]:
    for data_type in crop_cfg["apply_to"]:
        for item in data_by_type[data_type]:
            item["input"] = item["input"][:input_max_length]
            item["output"] = item["output"][:output_max_length]

# Make sure the output base directory exists
os.makedirs(output_dataset_name, exist_ok=True)

json_save_list(train_data, filepath=join(DATASET_DIR, output_dataset_name, "train.json"))
json_save_list(valid_data, filepath=join(DATASET_DIR, output_dataset_name, "valid.json"))

# For the test data we consider different processing.
for filename, data in test_data.items():
    json_save_list(data, filepath=join(DATASET_DIR, output_dataset_name, f"test_{filename.split('_')[-1]}"))
