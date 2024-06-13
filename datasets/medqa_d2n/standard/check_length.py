import json

# Load the modified data
with open('medqa_d2n_train.json', 'r') as file:
    data = json.load(file)

# Initialize variables to track the output with the most words
max_word_count = 0
output_with_most_words = ""

# Iterate through each entry in the dataset to find the output with the most words
for entry in data:
    # Split the output into words and count them
    word_count = len(entry["input"].split())
    if word_count > max_word_count:
        max_word_count = word_count
        output_with_most_words = entry["input"]

# Print the highest word count and the corresponding output
print("The highest word count is:", max_word_count)
print("The corresponding output is:", output_with_most_words)
