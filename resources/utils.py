import json
import random
from os.path import dirname, realpath

DATASETS_DIR = dirname(realpath(__file__))

EXTRACT_PROMPT = 'Extract the key information from clinical text'


def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_write(dict_iter, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('[')
        first = True
        for item in dict_iter:
            if not first:
                f.write(',\n')
            else:
                first = False
            json.dump(item, f, indent=2)
        f.write(']')


def json_save_list(valid_data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2)


def drop_column(data, column_name):
    for i in data:
        del i[column_name]
    return data


def split_dataset(json_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):

    # Check ratio sum
    total = train_ratio + valid_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Got: {total}")

    # Load data
    data = load_data(json_path)

    # Shuffle data
    rng = random.Random(seed)
    rng.shuffle(data)

    # Compute split indices
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    # Split data
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    return train_data, valid_data, test_data
