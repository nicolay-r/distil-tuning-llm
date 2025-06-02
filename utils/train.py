import shutil
import time
from datetime import datetime
from os.path import join

import wandb
import os
import logging
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
        seed=run,
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
        save_total_limit=1,                             # When save_total_limit = 1 and load_best_model_at_end
        load_best_model_at_end=True,                    # it is possible that two checkpoints are saved:
                                                        # the last one and the best one (if they are different).
        logging_steps=10,
        report_to="none",
        #######################################################################################################
        # Evaluation.
        #######################################################################################################
        metric_for_best_model="eval_rouge_avg",
        eval_delay=1,                                           # We don't want to start with evaluation.
        eval_strategy='steps',
        eval_steps=args.save_and_eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,   # This parameter is critical due to
                                                                # implementation of the custom Rouge. operation.
        greater_is_better=True,
        #######################################################################################################
        # Model saving.
        #######################################################################################################
        save_steps=args.save_and_eval_steps,
        save_strategy='steps',
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,                         # Repo name.
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
                                 log_compute_loss_func=lambda data, step: wandb.log(data, step=step),
                                 log_pred_step_func=lambda data, step: wandb.log(data, step=step),
                                 **trainer_kwargs)
    elif args.model_type == 'standard':
        trainer = Trainer(**trainer_kwargs)

    set_wandb(trainer_kwargs, args)

    wandb.watch(model, log='gradients')

    trainer.train()
    wandb.finish()

    # Additionally push the original tokenizer into the hub.
    if args.hub_model_id is not None:
        tokenizer.push_to_hub(args.hub_model_id)
    