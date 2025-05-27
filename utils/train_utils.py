# Copyright 2023 The Distilling-step-by-step authors

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

import wandb
import os
import logging
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
import torch

from utils.trainer_utils import TaskPrefixDataCollator, TaskPrefixTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_config_dir(args):
    path = f'{args.model_type}/{args.from_pretrained.split("/")[1]}_{args.addi_info}'
    return path


def set_wandb(trainer_kwargs, args):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow",
               project="MeDistill",
               mode="disabled",
               name=f"fine-tuning-{args.addi_info}-{dt_object}",
               config=trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.from_pretrained)
   
    config_dir = get_config_dir(args)
    output_dir = f'../ckpts/{config_dir}'  # for model ckpts
    # logging_dir = f'logs/{config_dir}'  # for training logs
    print("output dir: {}".format(output_dir))
   
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        os.removedirs(output_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500, # 预热步数
        num_training_steps=1201 * args.train_epochs
    )
    optimizers = (optimizer, scheduler)

    kwargs = {}
    if args.deepspeed is not None:
        kwargs["deepspeed"] = args.deepspeed

    training_args = Seq2SeqTrainingArguments(
        output_dir,                                         # 输出目录，模型和训练日志将被保存在这里
        weight_decay=0.01,
        eval_delay=1,
        num_train_epochs=args.train_epochs,
        report_to = "none",
        remove_unused_columns = False,                      # 是否移除未使用的列，默认为False，即保留所有列
        eval_strategy='steps',                              # 评估策略，这里设置为“steps”，表示按步数进行评估
        eval_steps=args.eval_steps,                         # 每隔多少步进行一次评估
        save_strategy='steps',                              # 保存策略
        save_steps=args.eval_steps,                         # 每隔多少步保存一次模型
        logging_steps=1,                                    # 每隔多少步记录一次日志
        learning_rate=args.lr,                              # 学习率
        warmup_steps=500,
        gradient_accumulation_steps=args.grad_steps,        # 梯度累积步数，用于实现更大的有效批大小
        per_device_train_batch_size=args.batch_size_train,  # 每个设备上的训练批大小
        per_device_eval_batch_size=args.batch_size_eval,    # 每个设备上的评估批大小
        predict_with_generate=True,                         # 是否使用生成模式进行预测
        seed=run,                                           # 随机种子，用于确保结果可复现
        local_rank=args.local_rank,                         # 本地排名，用于分布式训练
        bf16=args.bf16,                                     # 是否使用bfloat16进行训练，这可以提高性能
        generation_max_length=args.gen_max_len,             # 生成的最大长度
        prediction_loss_only=False,                         # 是否只预测损失，这里设置为False
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="test_rouge_avg",
        greater_is_better=True,
        push_to_hub=False,
        # optimizers=(optimizer, scheduler),                # 注意这里传递的是一个元组(optimizer, scheduler)
        **kwargs
    )

    if args.model_type == 'standard':
        print("model_type: {}".format(args.model_type))
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    else:
        print("model_type: {}".format(args.model_type))
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

    trainer_kwargs = {
        'alpha': args.alpha,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["valid"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,  
    }

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
        trainer.optimizers = optimizers

    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer = Seq2SeqTrainer(**trainer_kwargs)

    else:
        raise ValueError
    
    set_wandb(trainer_kwargs, args)

    wandb.watch(model, log='gradients')

    trainer.train()
    wandb.finish()
    