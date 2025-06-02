import argparse

from cfg import ROOT_DIR, SUMMARIZE_PROMPT
from resources.utils import EXTRACT_PROMPT
from utils.multiclinsum_loader import MultiClinSumDatasetLoader
from utils.train import train_and_evaluate
from transformers import AutoTokenizer


def assistant_prompt(instruction, text, summary):
    return (
        "<|im_start|>user\n"
        f"{instruction}:\n{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{summary}\n"
        "<|im_end|>"
    )


def tokenizer_func(tokenizer, max_length):
    return lambda text: tokenizer(text, max_length=max_length, padding="max_length", truncation=True)


def instruction_tokenizer_func(tokenizer_func, prompt, field_mapping_dict=None):

    # Tokenize full sequence.
    tokenized = tokenizer_func(prompt)

    # Find where assistant starts to compute label masking.
    assistant_start = prompt.find("<|im_start|>assistant")
    prompt_prefix = prompt[:assistant_start]
    prefix_ids = tokenizer_func(prompt_prefix)["input_ids"]
    prefix_len = len(prefix_ids)

    # Create labels
    labels = tokenized["input_ids"].copy()
    # Mask prompt part.
    labels[:prefix_len] = [-100] * prefix_len

    data = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

    return data if field_mapping_dict is None else \
        {new_name: data[origin_name] for origin_name, new_name in field_mapping_dict.items()}


def run(args):

    dataset_loader = MultiClinSumDatasetLoader(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    # Compose prompts for inputs.
    datasets = dataset_loader.load_from_json().map(
        lambda record: {
            "summarization_task": assistant_prompt(instruction=SUMMARIZE_PROMPT, text=record["input"], summary=record["output"])
        }
    )

    tok_func = tokenizer_func(tokenizer, max_length=args.max_input_length)

    # Process the input and output fields that are common for both modes.
    tokenized = datasets.map(
        lambda record: instruction_tokenizer_func(tokenizer_func=tok_func,
                                                  prompt=record["summarization_task"]),
        remove_columns=["summarization_task"]
    )

    if args.model_type == "standard":
        # For the standard task we do not support rationale.
        tokenized = tokenized.remove_columns([
            "rationale", "input", "output"
        ])

    if args.model_type == "distill":
        # 1. Compose new input for explanations and move rationale to output.
        # 2. Map this new input onto "input_ids_expl" and "attention_mask_expl", and "labels_expl.
        tokenized = tokenized.map(
            lambda record: {
                "rationale_task": assistant_prompt(instruction=EXTRACT_PROMPT,
                                                   text=record["input"],
                                                   summary=record["rationale"])
            },
            remove_columns=["input", "output", "rationale"]
        ).map(
            lambda record: instruction_tokenizer_func(tokenizer_func=tok_func,
                                                      prompt=record["rationale_task"],
                                                      field_mapping_dict={
                                                          "input_ids": "input_ids_expl",
                                                          "attention_mask": "attention_mask_expl",
                                                          "labels": "labels_expl"
                                                      }),
            remove_columns=["rationale_task"]
        )

    train_and_evaluate(args=args, run=args.seed, tokenizer=tokenizer,
                       tokenized_datasets=tokenized.map(batched=True),
                       root_dir=ROOT_DIR)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='distill', choices=["standard", "distill"])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--save_and_eval_steps', type=int, default=None)
    parser.add_argument('--eval_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=64)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--description', type=str, default="")
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--hub_model_id', type=str, default=None)

    args = parser.parse_args()

    run(args)