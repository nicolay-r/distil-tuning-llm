import json
from os.path import join
from statistics import mean, median

from cfg import ROOT_DIR


def analyze_input_lengths_in_chars(json_path, entry_value):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lengths = []
    for entry in data:
        input_text = entry.get(entry_value, "")
        if isinstance(input_text, str):
            lengths.append(len(input_text))

    if not lengths:
        return {"message": "No valid 'input' fields found."}

    sorted_lengths = sorted(lengths)

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 2),
        "median": median(lengths),
        "percentile_90": sorted_lengths[int(len(sorted_lengths) * 0.9)],
        "percentile_95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
    }


dataset_dir = join(ROOT_DIR, "datasets/multiclinsum_mult/")

for split in ["train", "test", "valid"]:

    for entry in ["input", "output"]:

        log = "\t".join([
            split,
            entry,
            str(analyze_input_lengths_in_chars(json_path=join(dataset_dir, split + ".json"), entry_value=entry))
            ])

        print(log)