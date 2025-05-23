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
import re
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from utils.head_utils import T5WithMLPHead
import torch

from utils.trainer_utils import TaskPrefixDataCollator, TaskPrefixTrainer, TaskPrefixDataCollator_hierarchical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config_dir(args):
    path = f'{args.model_type}/{args.from_pretrained.split("/")[1]}_{args.addi_info}'
    return path

def set_wandb(trainer_kwargs, args):
    # 获取当前时间的时间戳
    timestamp = time.time()

    # 将时间戳转换为datetime对象
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow", project="MeDistill", name=f"fine-tuning-{args.addi_info}-{dt_object}", config=trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)
#     breakpoint()
    if args.model_type == 'peft':
        from peft import get_peft_model
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
        if args.peft_type == 'adalora':
            from peft import AdaLoraConfig, PeftConfig, TaskType, get_peft_model  
            # 微调所有线性层
            pattern = r'\((\w+)\): Linear'
            linear_layers = re.findall(pattern, str(model.modules))
            target_modules = list(set(linear_layers))
            # ['q', 'wi_0', 'lm_head', 'k', 'v', 'o', 'wi_1', 'wo']

            peft_config = AdaLoraConfig(
                # peft_type="ADALORA",
                # r=2,
                init_r=4,                           #初始压缩率，表示在训练开始时模型参数的压缩程度。
                target_r=2,                         # 目标压缩率，表示训练结束时希望达到的模型参数的压缩程度。
                beta1=0.85,                         # 优化器的第一个动量参数，常用于计算梯度的指数衰减平均，有助于稳定训练过程。
                beta2=0.85,                         # 优化器的第二个动量参数，用于计算梯度平方的指数衰减平均，通常用于自适应学习率算法。
                tinit=20,                           # 训练开始阶段，初始阶段的训练时长或迭代次数。
                tfinal=1000,                        # nvidi训练结束阶段，最终阶段的训练时长或迭代次数。
                deltaT=10,                          # 每隔多少训练步骤更新一次压缩率，用于控制压缩率变化的频率。
                lora_alpha=32,                      # LoRA扩展的秩数，控制了低秩适应中秩的大小，影响模型调整的幅度。
                lora_dropout=0.05,                  # LoRA层中的dropout比率，用于防止过拟合，增加模型的泛化能力。
                task_type=TaskType.SEQ_2_SEQ_LM,    # 指定任务类型为序列到序列的语言模型，适用于需要生成序列输出的任务，如文本摘要。
                inference_mode=False,               # 指定是否为推理模式，False表示当前配置是用于训练。在推理时通常需要改为True。
                target_modules=target_modules,
            )       
        elif args.peft_type =="prefix":
            from peft import get_peft_model, PrefixEncoder, PrefixTuningConfig

            peft_config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING",
                task_type="SEQ_2_SEQ_LM",
                num_virtual_tokens=20,
            )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # breakpoint()
    elif args.model_type == 'task_prefix':
        if args.with_head:
            model = T5WithMLPHead.from_pretrained(args.from_pretrained).to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

        
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符
   
    # 整理路径
    config_dir = get_config_dir(args)
    output_dir = f'../ckpts/{config_dir}'  # for model ckpts
    # logging_dir = f'logs/{config_dir}'  # for training logs
    print("output dir: {}".format(output_dir))
   
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        os.removedirs(output_dir)
    # 路径整理完了
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 定义调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500, # 预热步数
        num_training_steps=1201 * args.train_epochs
    )
    optimizers = (optimizer, scheduler)
      

    # 设置一些训练中的细节参数 -- step --
    training_args = Seq2SeqTrainingArguments(
        output_dir,                         # 输出目录，模型和训练日志将被保存在这里
        weight_decay=0.01,
        eval_delay=1,
        num_train_epochs=args.train_epochs,
        report_to = "none",
        remove_unused_columns = False,      # 是否移除未使用的列，默认为False，即保留所有列
        eval_strategy='steps',            # 评估策略，这里设置为“steps”，表示按步数进行评估
        eval_steps=args.eval_steps,         # 每隔多少步进行一次评估
        save_strategy='steps',                 # 保存策略
        save_steps=args.eval_steps,         # 每隔多少步保存一次模型
        logging_steps=1,      # 每隔多少步记录一次日志
        learning_rate=args.lr,              # 学习率
        warmup_steps=500,
        gradient_accumulation_steps=args.grad_steps,  # 梯度累积步数，用于实现更大的有效批大小
        per_device_train_batch_size=args.batch_size_train,  # 每个设备上的训练批大小
        per_device_eval_batch_size=args.batch_size_eval,   # 每个设备上的评估批大小
        predict_with_generate=True,         # 是否使用生成模式进行预测
        seed=run,                           # 随机种子，用于确保结果可复现
        local_rank=args.local_rank,         # 本地排名，用于分布式训练
        bf16=args.bf16,                     # 是否使用bfloat16进行训练，这可以提高性能
        generation_max_length=args.gen_max_len,      # 生成的最大长度
        prediction_loss_only=False,         # 是否只预测损失，这里设置为False
        deepspeed=args.deepspeed,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="test_rouge_avg",
        greater_is_better=True,
        push_to_hub=False,
        # optimizers=(optimizer, scheduler),  # 注意这里传递的是一个元组(optimizer, scheduler)
    )
    # breakpoint()

    if args.model_type == 'standard':
        print("model_type: {}".format(args.model_type))
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    else:
        if args.hierarchical:
            data_collator = TaskPrefixDataCollator_hierarchical(tokenizer=tokenizer, model=model)
        else:
            print("model_type: {}".format(args.model_type))
            # rouge_metric = datasets.load_metric("rouge")
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

    # breakpoint()
    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
        trainer.optimizers = optimizers

    elif args.model_type == 'peft':
        # trainer = AdptTrainer(**trainer_kwargs)
        pass

    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        
        trainer = Seq2SeqTrainer(**trainer_kwargs) # Seq2SeqTrainer是Hugging Face Transformers库中的一个类，专门用于序列到序列（sequence-to-sequence）的模型训练，比如T5、BART等。
        '''解释一下：训练的是T5模型，而Seq2SeqTrainer是用于训练过程的工具。'''
        # breakpoint()
        
    else:
        raise ValueError
    
    set_wandb(trainer_kwargs, args)

    wandb.watch(model, log = 'gradients')

    trainer.train()
    wandb.finish()
    