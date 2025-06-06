import argparse
import time
from datetime import datetime

import wandb

from cfg import ROOT_DIR, SUMMARIZE_PROMPT
from resources.utils import EXTRACT_PROMPT
from utils.multiclinsum_loader import MultiClinSumDatasetLoader
from transformers import AutoTokenizer

import shutil
from os.path import join

import os
import logging
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.trainer_utils import set_seed

from utils.distill_collator import DistillDataCollator
from utils.distill_trainer import DistillTrainer
from utils.metrics import compute_metrics_rouge


def set_wandb(trainer_kwargs, description):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow",
               project="Distill-LM",
               #mode="disabled",
               name=f"fine-tuning-{description}-{dt_object}",
               config=trainer_kwargs)


def train_and_evaluate(args, tokenizer, tokenized_datasets):
    """ This is the main code for training and evaluation.
    """

    set_seed(args.seed)

    # Initialize model with the related configuration parameters.
    model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)
    model.generation_config.max_length = args.max_output_length

    output_dir = join(
        ROOT_DIR,
        '.ckpts',
        args.model_type,
        args.from_pretrained.split("/")[-1] + "_" + args.description
    )

    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir,
        seed=args.seed,
        ######################################################################################################
        # Scheduler:
        # According to: https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune
        ######################################################################################################
        num_train_epochs=args.train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        learning_rate=args.lr,
        optim="adamw_torch",
        #######################################################################################################
        # Memory
        #######################################################################################################
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_eval,
        bf16=args.bf16,
        remove_unused_columns=False,
        #######################################################################################################
        # Logging
        #######################################################################################################
        save_total_limit=1,  # When save_total_limit = 1 and load_best_model_at_end
        load_best_model_at_end=True,  # it is possible that two checkpoints are saved:
        # the last one and the best one (if they are different).
        logging_steps=10,
        report_to="none",
        #######################################################################################################
        # Evaluation.
        #######################################################################################################
        metric_for_best_model="eval_rouge_avg",
        eval_delay=1,  # We don't want to start with evaluation.
        eval_strategy='steps',
        eval_steps=args.save_and_eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,  # This parameter is critical due to
        # implementation of the custom Rouge. operation.
        greater_is_better=True,
        #######################################################################################################
        # Model saving.
        #######################################################################################################
        save_steps=args.save_and_eval_steps,
        save_strategy='steps',
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,  # Repo name.
    )

    print("model_type: {}".format(args.model_type))

    if args.model_type == 'distill':
        data_collator = DistillDataCollator(tokenizer=tokenizer, mlm=False)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': tokenized_datasets["valid"],
        'data_collator': data_collator,
        'compute_metrics': lambda eval_preds: compute_metrics_rouge(eval_preds=eval_preds, tokenizer=tokenizer),
    }

    if args.model_type == 'distill':
        trainer = DistillTrainer(alpha=args.alpha,
                                 log_compute_loss_func=lambda data: wandb.log(data),
                                 log_pred_step_func=lambda data: wandb.log(data),
                                 **trainer_kwargs)
    elif args.model_type == 'standard':
        trainer = Trainer(**trainer_kwargs)

    set_wandb(trainer_kwargs=trainer_kwargs, description=args.description)

    wandb.watch(model, log='gradients')

    trainer.train()
    wandb.finish()

    # Additionally push the original tokenizer into the hub.
    if args.hub_model_id is not None:
        tokenizer.push_to_hub(args.hub_model_id)


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

    train_and_evaluate(
        args=args,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized.map(batched=True)
    )


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
    parser.add_argument('--train_epochs', type=int, default=None)
    parser.add_argument('--hub_model_id', type=str, default=None)

    args = parser.parse_args()

    run(args)