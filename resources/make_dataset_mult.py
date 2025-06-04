import sys
sys.path.append("..")

import os
from os.path import join
from cfg import DATASET_DIR
from utils import split_dataset, json_save_list, drop_column

#############################################
# Initial parameters for setting up datasets.
#############################################
input_dataset_name = "multiclinsum_rationale"
output_dataset_name = "multiclinsum_rationale_mult"
input_files = [
    "multiclinsum_gs_en.json",
    "multiclinsum_gs_es.json",
    "multiclinsum_gs_fr.json",
    "multiclinsum_gs_pt.json"
]
input_max_length = 2560
output_max_length = 512
train_ratio = 0.8
# We make validation ratio too small due to unknown memory leakage on resource consumption during evaluation.
valid_ratio = 0.01
test_ratio = 0.19

train_data = []
valid_data = []
test_data = {}      # We do not merge test data.

for filename in input_files:

    train, valid, test = split_dataset(
        json_path=join(DATASET_DIR, input_dataset_name, filename),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
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
    drop_column(data, column_name="rationale")             # For the test data we also dropping `rationale`.

# Crop data.
for data in [train_data, valid_data]:
    for item in data:
        item["input"] = item["input"][:input_max_length]
        item["output"] = item["output"][:output_max_length]

# Make sure the output base directory exists
os.makedirs(output_dataset_name, exist_ok=True)

json_save_list(train_data, filepath=join(DATASET_DIR, output_dataset_name, "train.json"))
json_save_list(valid_data, filepath=join(DATASET_DIR, output_dataset_name, "valid.json"))

# For the test data we consider different processing.
for filename, data in test_data.items():
    json_save_list(test_data, filepath=join(DATASET_DIR, output_dataset_name, f"test_{filename}"))
