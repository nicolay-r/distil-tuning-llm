import json

# Load the JSON data from the file
with open('medqa_d2n_valid.json', 'r') as file:
    data = json.load(file)

unique_keys = set()

# Loop through each entry and collect keys from the 'rationale' part
for entry in data:
    unique_keys.update(entry['rationale'].keys())

# Print all unique keys found in the 'rationale' dictionaries
print(unique_keys)
