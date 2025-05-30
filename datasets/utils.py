import json
import random


def split_dataset(json_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):

    # Check ratio sum
    total = train_ratio + valid_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Got: {total}")

    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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


    print(f"Dataset split into: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test.")
