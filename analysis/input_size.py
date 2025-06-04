import json
from os.path import join
from statistics import mean, median

from cfg import DATASET_DIR


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


preset_name = "multiclinsum_rationale_mult_test"

datasets_meta = {
    "multiclinsum": {
        "splits": ["multiclinsum_gs_en", "multiclinsum_gs_es", "multiclinsum_gs_fr", "multiclinsum_gs_pt"],
        "cols": ["input", "output"]
    },
    "multiclinsum_rationale": {
        "splits": ["multiclinsum_gs_en", "multiclinsum_gs_es", "multiclinsum_gs_fr", "multiclinsum_gs_pt"],
        "cols": ["rationale"]
    },
    "multiclinsum_rationale_mult": {
        "splits": ["train", "valid", ],
        "cols": ["input", "output", "rationale"]
    },
    "multiclinsum_rationale_mult_test": {
        "dir": "multiclinsum_rationale_mult",
        "splits": ["test_en", "test_pt", "test_fr", "test_es"],
        "cols": ["input", "output"]
    }
}

meta = datasets_meta[preset_name]

for split in meta["splits"]:
    dataset_name = meta.get("dir", split)
    for entry in meta["cols"]:
        filepath = join(DATASET_DIR, dataset_name, split + ".json")
        log = "\t".join([
            dataset_name,
            entry,
            str(analyze_input_lengths_in_chars(json_path=filepath, entry_value=entry))
            ])
        print(log)