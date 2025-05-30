# Copyright 2023 The Distilling-step-by-step authors
import shutil
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from datetime import datetime
from os.path import dirname, realpath, join

import wandb
import os
import logging
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, \
    get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
import torch

from utils.metrics import compute_metrics_rouge
from utils.trainer_utils import TaskPrefixDataCollator, TaskPrefixTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_config_dir(args):
    path = f'{args.model_type}/{args.from_pretrained.split("/")[1]}_{args.addi_info}'
    return path


def set_wandb(trainer_kwargs, args):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow",
               project="MeDistill-d2n-long",
               mode="disabled",
               name=f"fine-tuning-{args.addi_info}-{dt_object}",
               config=trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets):
    set_seed(run)

    model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)

    # Set maximum generation length.
    model.config.max_length = args.max_output_length

    config_dir = get_config_dir(args)

    current_dir = dirname(realpath(__file__))
    output_dir = join(current_dir, f'.ckpts/{config_dir}')
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
        local_rank=args.local_rank,
        bf16=args.bf16,
        prediction_loss_only=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="test_rouge_avg",
        greater_is_better=True,
        push_to_hub=False,
        # IMPORTANT.
        # This parameter is critical due to implementation of the custom Rouge. operation.
        eval_accumulation_steps=5,
    )

    print("model_type: {}".format(args.model_type))

    if args.model_type == 'standard':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    else:
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {
            'test': tokenized_datasets["valid"]
        },
        'data_collator': data_collator,
        'compute_metrics': lambda eval_preds: compute_metrics_rouge(eval_preds=eval_preds, tokenizer=tokenizer),
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(alpha=args.alpha, **trainer_kwargs)

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
    