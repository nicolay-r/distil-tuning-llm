import json
import os
import random
from os import listdir
from os.path import join, isfile

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


def iter_text_files(folder_path, encoding="utf-8", max_content_length=None,
                    skip_if_exists_in=None, fmt_filename_func=lambda fp: fp):
    for filename in listdir(folder_path):
        source_path = join(folder_path, filename)

        if not os.path.isfile(source_path):
            continue

        if skip_if_exists_in:
            target_path = os.path.join(skip_if_exists_in, fmt_filename_func(filename))
            if os.path.exists(target_path):
                print("OK [SKIP]", target_path)
                continue

        try:
            with open(source_path, 'r', encoding=encoding) as f:
                content = f.read()

                if max_content_length is not None:
                    content = content[:max_content_length]

                yield filename, content
        except (UnicodeDecodeError, OSError):
            continue


def write_text_files(file_iter, folder_path, encoding="utf-8"):
    os.makedirs(folder_path, exist_ok=True)
    for filename, content in file_iter:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
        except OSError as e:
            print(f"Error writing to {filepath}: {e}")


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
