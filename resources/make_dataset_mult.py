import os
from os.path import join

from utils import split_dataset, DATASETS_DIR, json_save_list, drop_column

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
test_data = []

for filename in input_files:

    train, valid, test = split_dataset(
        json_path=join(DATASETS_DIR, input_dataset_name, filename),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
    )

    train_data += train
    valid_data += valid
    test_data += test

# Remove non utilized strings.
drop_column(train_data, column_name="rationale_prompt")
drop_column(valid_data, column_name="rationale_prompt")
drop_column(test_data, column_name="rationale_prompt")

# Crop data.
for data in [train_data, valid_data, test_data]:
    for item in data:
        item["input"] = item["input"][:input_max_length]
        item["output"] = item["output"][:output_max_length]
        item["rationale"] = item["rationale"][:output_max_length]

# Make sure the output base directory exists
os.makedirs(output_dataset_name, exist_ok=True)

json_save_list(train_data, filepath=join(DATASETS_DIR, output_dataset_name, "train.json"))
json_save_list(valid_data, filepath=join(DATASETS_DIR, output_dataset_name, "valid.json"))
json_save_list(test_data, filepath=join(DATASETS_DIR, output_dataset_name, "test.json"))
