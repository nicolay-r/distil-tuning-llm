import json

# Load the JSON data from the file
with open('medqa_d2n_valid.json', 'r') as file:
    data = json.load(file)

# Loop through each entry
for entry in data:
    # Access the 'rationale' dictionary and replace None with "No"
    for key, value in entry['rationale'].items():
        if value == 'None':
            entry['rationale'][key] = "No"

# Save the updated data back to the file if needed
with open('medqa_d2n_valid.json', 'w') as file:
    json.dump(data, file, indent=4)

# Optionally print out the updated data to verify changes

