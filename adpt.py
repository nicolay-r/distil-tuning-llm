from adapters import AutoAdapterModel
import adapters
import adapters.composition as ac

from datasets import load_dataset
from adapters.composition import Stack

dataset_en = load_dataset("super_glue", "copa")
# dataset_en.num_rows

model = AutoAdapterModel.from_pretrained("google/flan-t5-small")
# adapters.init(model)

# model.add_adapter("xiaoxiao_adapter_a")
# model.add_adapter("xiaoxiao_adapter_b")
# model.active_adapters = ac.Stack("xiaoxiao_adapter_a", "xiaoxiao_adapter_b")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def encode_batch(examples):
  """Encodes a batch of input data using the model tokenizer."""
  all_encoded = {"input_ids": [], "attention_mask": []}
  # Iterate through all examples in this batch
  for premise, question, choice1, choice2 in zip(examples["premise"], examples["question"], examples["choice1"], examples["choice2"]):
    sentences_a = [premise + " " + question for _ in range(2)]
    # Both answer choices are passed in an array according to the format needed for the multiple-choice prediction head
    sentences_b = [choice1, choice2]
    encoded = tokenizer(
        sentences_a,
        sentences_b,
        max_length=60,
        truncation=True,
        padding="max_length",
    )
    all_encoded["input_ids"].append(encoded["input_ids"])
    all_encoded["attention_mask"].append(encoded["attention_mask"])
  return all_encoded

def preprocess_dataset(dataset):
  # Encode the input data
  dataset = dataset.map(encode_batch, batched=True)
  # The transformers model expects the target class column to be named "labels"
  dataset = dataset.rename_column("label", "labels")
  # Transform to pytorch tensors and only output the required columns
  dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
  return dataset

dataset_en = preprocess_dataset(dataset_en)
print(dataset_en)

model.add_adapter("xiaoxiao_adapter_a")
model.train_adapter(["xiaoxiao_adapter_a"])