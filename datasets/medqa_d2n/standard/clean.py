import json

# Step 1: Load the data
with open('medqa_d2n_valid1.json', 'r') as file:
    data = json.load(file)

# Step 2: Modify the data
for entry in data:
    original_output = entry["output"]
    rationale = entry["rationale"]
    # Formatting the output as a single string with both parts labeled
    formatted_output = f"Summarization: {original_output}. Rationale: {rationale}."
    entry["output"] = formatted_output
    # Optionally, remove the old rationale key if you don't want to retain it in the main dictionary
    del entry["rationale"]

# Step 3: Save the modified data
with open('medqa_d2n_valid.json', 'w') as file:
    json.dump(data, file, indent=4)  # Use indent for pretty-printing if you like

print("Dataset has been modified and saved as 'medqa_d2n_train.json'")
