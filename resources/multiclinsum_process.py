import sys
sys.path.append("..")

import os
from cfg import DATASET_DIR
from os.path import join
from utils import split_dataset, json_save_list, drop_column, load_data


# Load configuration
config = load_data(json_path="cfg_multiclinsum2025.json")

data_cfg = config["dataset_config"]
processing_cfg = config["processing_config"]

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

# Remove non utilized strings based on config
drop_columns_cfg = processing_cfg["drop_columns"]

# Drop columns.
for data_type in ["train", "valid"]:
    if data_type in drop_columns_cfg:
        for column_name in drop_columns_cfg[data_type]:
            drop_column(data_by_type[data_type], column_name=column_name)

# Drop columns for test data (handled individually for each subtask)
if "test" in drop_columns_cfg:
    for data in test_data.values():
        for column_name in drop_columns_cfg["test"]:
            drop_column(data, column_name=column_name)

# Crop data based on config.
crop_cfg = processing_cfg["crop_data"]
if crop_cfg["enabled"]:
    for data_type in crop_cfg["apply_to"]:
        for item in data_by_type[data_type]:
            item["input"] = item["input"][:input_max_length]
            item["output"] = item["output"][:output_max_length]

output_dir = join(DATASET_DIR, output_dataset_name)

# Make sure the output base directory exists
os.makedirs(output_dir, exist_ok=True)

# Writing training and validation data.
for split, data in data_by_type.items():
    json_save_list(data, filepath=join(output_dir, f"{split}.json"))

# For the test data we consider different processing.
for filename, data in test_data.items():
    json_save_list(data, filepath=join(output_dir, f"test_{filename.split('_')[-1]}"))
