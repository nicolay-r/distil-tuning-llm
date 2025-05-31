import shutil
import time
from datetime import datetime
from os.path import join

import torch
import wandb
import os
import logging
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, \
    get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from utils.distill_collator import DistillDataCollator
from utils.distill_trainer import DistillTrainer
from utils.metrics import compute_metrics_rouge


def set_wandb(trainer_kwargs, args):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow",
               project="Distill-LM",
               #mode="disabled",
               name=f"fine-tuning-{args.description}-{dt_object}",
               config=trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, root_dir):
    set_seed(run)

    # Initialize model with the related configuration parameters.
    model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)
    model.generation_config.max_length = args.max_output_length

    output_dir = join(
        root_dir,
        '.ckpts',
        args.model_type,
        args.from_pretrained.split("/")[-1] + "_" + args.description
    )

    print("output dir: {}".format(output_dir))

    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir,
        weight_decay=0.01,
        eval_delay=1,
        num_train_epochs=args.train_epochs,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.eval_steps,
        logging_steps=1,
        learning_rate=args.lr,
        warmup_steps=500,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_eval,
        seed=run,
        bf16=args.bf16,
        prediction_loss_only=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge_avg",
        greater_is_better=True,
        push_to_hub=False,
        # IMPORTANT.
        # This parameter is critical due to implementation of the custom Rouge. operation.
        eval_accumulation_steps=args.eval_accumulation_steps,
    )

    print("model_type: {}".format(args.model_type))

    if args.model_type == 'standard':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    else:
        data_collator = DistillDataCollator(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': tokenized_datasets["valid"],
        'data_collator': data_collator,
        'compute_metrics': lambda eval_preds: compute_metrics_rouge(eval_preds=eval_preds, tokenizer=tokenizer),
    }

    if args.model_type == 'distill':
        trainer = DistillTrainer(
            alpha=args.alpha,
            log_compute_loss_func=lambda data: wandb.log(data),
            log_pred_step_func=lambda data, step: wandb.log(data, step=step),
            **trainer_kwargs
        )

    elif args.model_type == 'standard':
        trainer = Trainer(**trainer_kwargs)

    # Setup optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=1201 * args.train_epochs
    )
    optimizers = (optimizer, scheduler)
    trainer.optimizers = optimizers

    set_wandb(trainer_kwargs, args)

    wandb.watch(model, log='gradients')

    trainer.train()
    wandb.finish()
    