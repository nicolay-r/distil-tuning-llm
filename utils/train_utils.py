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

# from adapters import Seq2SeqAdapterTrainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from utils.data_utils import MEDQADatasetLoader

<<<<<<< HEAD
from utils.trainer_utils import TaskPrefixDataCollator, TaskPrefix_COS, CoTTrainer, AdptTrainer
=======
from utils.trainer_utils import TaskPrefixDataCollator, TaskPrefixTrainer, TaskPrefix_COS, CoTTrainer, AdptTrainer,TaskPrefixTrainerWithHead
>>>>>>> 9bea8eaeeeb0d1d4d3be08fe69a3719c4d1e914e


def get_config_dir(args):
    
    path = f'{args.model_type}/{args.from_pretrained.split("/")[1]}_{args.addi_info}'
    return path

def set_wandb(trainer_kwargs):
    # 获取当前时间的时间戳
    timestamp = time.time()

    # 将时间戳转换为datetime对象
    dt_object = datetime.fromtimestamp(timestamp)
    wandb.init(group="lmflow", project="new-sota-model", name=f"fine-tuning-{dt_object}", config=trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)
    
    if args.model_type == 'adapter':
        # from adapters import AutoAdapterModel
        from transformers import AutoModelForSeq2SeqLM
        # from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PromptTuningConfig, TaskType
        from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftConfig
        from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

        
        '''
        fan_in_fan_out (bool)：是否将层的存储形式替换成 (fan_in, fan_out) 的样子，默认为False；
        bias (str)：是否添加偏置。参数选择为：[“none”,“all”,“lora_only”]。如果为"all"或"lora_only"，则相应的偏差将在训练期间更新；
        modules_to_save (List[str])：要在训练过程中保存的模块列表，默认为None 表示不保存；
        '''

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, 
            # target_modules (Union[List[str],str])：添加lora的目标模块名称；
            r=args.rank, # 低秩矩阵的维度，通常是1、2、4、8、16、32、64；
            lora_alpha=args.lora_alpha, # 缩放参数，控制低秩矩阵的适应程度——越小的值对模型参数进行压缩的越强烈，可能会影响模型性能；越大的值，则减轻了对模型参数的压缩，可能保留了更多的模型性能。不同的模型可能有不同的默认值和具体用法；
            lora_dropout=args.lora_dropout #防止过拟合的dropout；
            )
        
        # r=4: trainable params: 2,359,296 || all params: 2,852,116,480 || trainable%: 0.08272088522836206
        # r=8: trainable params: 4,718,592 || all params: 2,854,475,776 || trainable%: 0.16530502867367827
        # r=2: trainable params: 1,179,648 || all params: 2,850,936,832 || trainable%: 0.04137755655471499
        # r=1: trainable params: 589,824 || all params: 2,850,347,008 || trainable%: 0.0206930594185394

        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    else:
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符
    # breakpoint()
    if args.parallelize:
        model.parallelize() # 用于将 T5 模型的层分布到多个 GPU 上，以便并行处理。
    
    # 整理路径
    config_dir = get_config_dir(args)
    output_dir = f'../ckpts/{config_dir}'  # for model ckpts
    # logging_dir = f'logs/{config_dir}'  # for training logs
    print("output dir: {}".format(output_dir))
   
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        os.removedirs(output_dir)
    # 路径整理完了
    
    # 设置一些训练中的细节参数 -- step --
    training_args = Seq2SeqTrainingArguments(
        output_dir,                         # 输出目录，模型和训练日志将被保存在这里
        report_to = "none",
        remove_unused_columns = False,      # 是否移除未使用的列，默认为False，即保留所有列
        evaluation_strategy = 'steps',      # 评估策略，这里设置为“steps”，表示按步数进行评估
        eval_steps=args.eval_steps,         # 每隔多少步进行一次评估
        save_strategy='steps',                 # 保存策略
        save_steps=args.eval_steps,         # 每隔多少步保存一次模型
        logging_steps=1,      # 每隔多少步记录一次日志
        max_steps=args.max_steps,           # 最大步数，训练将在达到这个步数后停止
        learning_rate=args.lr,              # 学习率
        warmup_steps=1000,
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
        metric_for_best_model="test_accuracy",
        greater_is_better=True
    )
    # -- epoch --
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir,                         # 输出目录，模型和训练日志将被保存在这里
    #     report_to = "none",
    #     remove_unused_columns = False,      # 是否移除未使用的列，默认为False，即保留所有列
    #     evaluation_strategy = 'epoch',      # 评估策略，这里设置为“steps”，表示按步数进行评估
    #     num_train_epochs=5,
    #     # eval_steps=args.eval_steps,         # 每隔多少步进行一次评估
    #     save_strategy='epoch',                 # 保存策略
    #     save_steps=args.eval_steps,         # 每隔多少步保存一次模型
    #     logging_dir=logging_dir,            # 日志目录，训练日志将被保存在这里
    #     logging_strategy="epoch",  # 日志记录策略，目前是step
    #     logging_steps=1,      # 每隔多少步记录一次日志
    #     # max_steps=args.max_steps,           # 最大步数，训练将在达到这个步数后停止
    #     learning_rate=args.lr,              # 学习率
    #     warmup_steps=1000,
    #     gradient_accumulation_steps=args.grad_steps,  # 梯度累积步数，用于实现更大的有效批大小
    #     per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批大小
    #     per_device_eval_batch_size=args.batch_size,   # 每个设备上的评估批大小
    #     predict_with_generate=True,         # 是否使用生成模式进行预测
    #     seed=run,                           # 随机种子，用于确保结果可复现
    #     local_rank=args.local_rank,         # 本地排名，用于分布式训练
    #     bf16=args.bf16,                     # 是否使用bfloat16进行训练，这可以提高性能
    #     generation_max_length=args.gen_max_len,      # 生成的最大长度
    #     prediction_loss_only=False,         # 是否只预测损失，这里设置为False
    #     deepspeed=args.deepspeed,
    #     save_total_limit=1,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="test_accuracy",
    #     greater_is_better=True
    # )
    

    # if args.model_type == 'task_prefix':
    #     print("model_type: {}".format(args.model_type))
    #     # rouge_metric = datasets.load_metric("rouge")
    #     data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    # elif args.model_type == 'CoT':
    #     print("model_type: {}".format(args.model_type))
    #     data_collator = CoTDataCollator(tokenizer=tokenizer, model=model)
    # elif args.model_type == 'adapter':
    #     print("model_type: {}".format(args.model_type))
    #     data_collator = AdapterDataCollator(tokenizer=tokenizer, model=model)
    if args.model_type == 'standard':
        print("model_type: {}".format(args.model_type))
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        print("model_type: {}".format(args.model_type))
        # rouge_metric = datasets.load_metric("rouge")
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

    
    
    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'weight': args.weight,
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
        if args.cos_sim:
            trainer = TaskPrefix_COS(**trainer_kwargs)
        elif args.with_head:
            trainer = TaskPrefixTrainerWithHead(**trainer_kwargs)
        else:
            trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'CoT':
        trainer = CoTTrainer(**trainer_kwargs)
        
    elif args.model_type == 'adapter':
        trainer = AdptTrainer(**trainer_kwargs)
        
    elif args.model_type == 'standard':
        
        trainer_kwargs.pop('alpha') # 从trainer_kwargs字典中删除键'alpha'及其对应的值。
        trainer_kwargs.pop('output_rationale')
        trainer_kwargs.pop('weight')
        trainer = Seq2SeqTrainer(**trainer_kwargs) # Seq2SeqTrainer是Hugging Face Transformers库中的一个类，专门用于序列到序列（sequence-to-sequence）的模型训练，比如T5、BART等。
        '''解释一下：训练的是T5模型，而Seq2SeqTrainer是用于训练过程的工具。'''
        # breakpoint()
        
    else:
        raise ValueError
    
    set_wandb(trainer_kwargs)

    wandb.watch(model, log = 'gradients')

    trainer.train()
    wandb.finish()
    
    # train_adapter()