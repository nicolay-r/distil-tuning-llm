import json
import os
from os.path import join, dirname, realpath

from utils import split_dataset


# List of input JSON files
input_files = [
    "multiclinsum_gs_en.json",
    "multiclinsum_gs_es.json",
    "multiclinsum_gs_fr.json",
    "multiclinsum_gs_pt.json"
]

# Split ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

train_data = []
valid_data = []
test_data = []

# Loop over each file and perform the split
current_dir = dirname(realpath(__file__))

for file in input_files:

    train, valid, test = split_dataset(
        json_path=join(current_dir, "multiclinsum", file),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
    )

    train_data += train
    valid_data += valid
    test_data += test


output_dir = "multiclinsum_mult"

# Make sure the output base directory exists
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2)
with open(f"{output_dir}/valid.json", 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, indent=2)
with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2)
