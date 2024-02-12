from adapters import AdapterSetup, AutoAdapterModel
# from adapters.models import T5ForCondiditionalGenerationWithHeadsMixin
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import PeftConfig



import adapters.composition as ac
from adapters.composition import Fuse
from transformers import T5Config
from adapters import T5AdapterModel, Seq2SeqAdapterTrainer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapters import AdapterTrainer
from datasets import load_dataset
data_files = {}
extension = "json"
data_files["train"] = "standard/medqa_d2n_train.json"
data_files["valid"] = "standard/medqa_d2n_valid.json"
text_column = 'input'
summary_column = "output"
prefix = "summarize:"
padding = False
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")


def preprocess_function(examples):
    # remove pairs where at least one record is None

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=128, padding=padding, truncation=True)


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_data = load_dataset(
            extension,
            data_files=data_files
        )
train_data =raw_data["train"]
valid_data = raw_data['valid']
column_names = raw_data["train"].column_names
train_dataset = train_data.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

eval_dataset = valid_data.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
# peft_config = PeftConfig.from_pretrained("./tmp/tst-summarization/summarization/")


model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
ckpt_path = "./tmp/tst-summarization/summarization/"
qc = model.load_adapter(adapter_state_dict=ckpt_path)
model.set_active_adapters(qc)
#
# checkpoint = torch.load(ckpt_path, map_location="cpu") #读取本地训练好的chekpoint
# model.load_state_dict(checkpoint)

# # Add a classification head for our target task
# model.add_classification_head("cb", num_labels=len(id2label))
# model.add_seq2seq_lm_head("sum_adpt")

# Unfreeze and activate fusion setup
# adapter_setup = Fuse(qc)
# model.train_adapter_fusion(adapter_setup)

training_args = Seq2SeqTrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy = "epoch",
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    predict_with_generate=True,
    fp16=True,
    remove_unused_columns=False,
    label_names=["output"]
)

from evaluate import load
import nltk

nltk.download("punkt")

metric = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': {'test': eval_dataset,},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
        # 'callbacks': [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    }

trainer = Seq2SeqTrainer(**trainer_kwargs)

trainer.train()

model.save_all_adapters("./saved")
