import json
import os
from os.path import join, dirname, realpath
from utils import split_dataset


#############################################
# Initial parameters for setting up datasets.
#############################################
dataset_name = "multiclinsum"
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

# Loop over each file and perform the split
current_dir = dirname(realpath(__file__))

for filename in input_files:

    train, valid, test = split_dataset(
        json_path=join(current_dir, dataset_name, filename),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
    )

    train_data += train
    valid_data += valid
    test_data += test


for data in [train_data, valid_data, test_data]:
    for item in data:
        item["input"] = item["input"][:input_max_length]
        item["output"] = item["output"][:output_max_length]
        item["rationale"] = item["rationale"][:output_max_length]


output_dir = "multiclinsum_mult"

# Make sure the output base directory exists
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2)
with open(f"{output_dir}/valid.json", 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=2)
with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2)
