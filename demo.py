from datasets import Dataset, DatasetDict, load_dataset

data_files = "datasets/svamp/svamp_train.json"
datasets = load_dataset('json', data_files=data_files)
datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)

# import pandas as pd
# from datasets import Dataset

# # 读取 JSON 文件
# data = pd.read_json("datasets/svamp/svamp_test.json")

# # 将 DataFrame 转换为 Hugging Face Dataset
# dataset = Dataset.from_pandas(data)


training_args = Seq2SeqTrainingArguments(
        output_dir,                         # 输出目录，模型和训练日志将被保存在这里
        remove_unused_columns = False,      # 是否移除未使用的列，默认为False，即保留所有列
        evaluation_strategy = 'steps',      # 评估策略，这里设置为“steps”，表示按步数进行评估
        eval_steps=args.eval_steps,         # 每隔多少步进行一次评估
        save_strategy='steps',                 # 保存策略，这里设置为“no”，表示不自动保存模型
        save_steps=args.eval_steps,         # 每隔多少步保存一次模型
        logging_dir=logging_dir,            # 日志目录，训练日志将被保存在这里
        logging_strategy=logging_strategy,  # 日志记录策略，目前是step
        logging_steps=args.eval_steps,      # 每隔多少步记录一次日志
        max_steps=args.max_steps,           # 最大步数，训练将在达到这个步数后停止
        learning_rate=args.lr,              # 学习率
        gradient_accumulation_steps=args.grad_steps,  # 梯度累积步数，用于实现更大的有效批大小
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批大小
        per_device_eval_batch_size=args.batch_size,   # 每个设备上的评估批大小
        predict_with_generate=True,         # 是否使用生成模式进行预测
        seed=run,                           # 随机种子，用于确保结果可复现
        local_rank=args.local_rank,         # 本地排名，用于分布式训练
        bf16=args.bf16,                     # 是否使用bfloat16进行训练，这可以提高性能
        generation_max_length=args.gen_max_len,      # 生成的最大长度
        prediction_loss_only=False,         # 是否只预测损失，这里设置为False
    )

trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

trainer = Seq2SeqTrainer(**trainer_kwargs)
trainer.train()



model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained) # args.from_pretrained通常是一个字符串，指向预训练模型的存储位置，可以是本地路径或者在线模型库的标识符

    if args.parallelize:
        model.parallelize() # 用于将 T5 模型的层分布到多个 GPU 上，以便并行处理。
    
    # 整理路径
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs
    print("output dir: {}".format(output_dir))
    print("log dir: {}".format(logging_dir))

    if args.no_log:
        # print("有")
        logging_strategy = 'no'
        logging_dir = None
    else:
        # 走的这里
        logging_strategy = 'steps'
   
    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)
    # 路径整理完了
    
    # training_args 主要用于设置训练过程的参数，如学习率、批大小、评估策略等，而不涉及模型的内部架构。
    training_args = Seq2SeqTrainingArguments(
        output_dir,                         # 输出目录，模型和训练日志将被保存在这里
        remove_unused_columns = False,      # 是否移除未使用的列，默认为False，即保留所有列
        evaluation_strategy = 'steps',      # 评估策略，这里设置为“steps”，表示按步数进行评估
        eval_steps=args.eval_steps,         # 每隔多少步进行一次评估
        save_strategy='steps',                 # 保存策略，这里设置为“no”，表示不自动保存模型
        save_steps=args.eval_steps,         # 每隔多少步保存一次模型
        logging_dir=logging_dir,            # 日志目录，训练日志将被保存在这里
        logging_strategy=logging_strategy,  # 日志记录策略，目前是step
        logging_steps=args.eval_steps,      # 每隔多少步记录一次日志
        max_steps=args.max_steps,           # 最大步数，训练将在达到这个步数后停止
        learning_rate=args.lr,              # 学习率
        gradient_accumulation_steps=args.grad_steps,  # 梯度累积步数，用于实现更大的有效批大小
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批大小
        per_device_eval_batch_size=args.batch_size,   # 每个设备上的评估批大小
        predict_with_generate=True,         # 是否使用生成模式进行预测
        seed=run,                           # 随机种子，用于确保结果可复现
        local_rank=args.local_rank,         # 本地排名，用于分布式训练
        bf16=args.bf16,                     # 是否使用bfloat16进行训练，这可以提高性能
        generation_max_length=args.gen_max_len,      # 生成的最大长度
        prediction_loss_only=False,         # 是否只预测损失，这里设置为False
    )

    

    if args.model_type == 'task_prefix':
        print("model_type: {}".format(args.model_type))
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        print("model_type: {}".format(args.model_type))
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError

    # 训练器的参数
    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model, 
        'args': training_args, # 模型的参数
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    
    '''
    为什么不直接将 training_args 传入 Seq2SeqTrainer 中？
    training_args 只包含与训练过程相关的配置参数，而 Seq2SeqTrainer 需要更多的信息来初始化训练过程，例如模型、数据集、数据整理器、标记器等。
    简而言之，T5是模型，training_args是模型的参数，Seq2SeqTrainer是训练器，trainer_kwargs是训练器的参数
    '''

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha') # 从trainer_kwargs字典中删除键'alpha'及其对应的值。
        trainer_kwargs.pop('output_rationale')
        # trainer_kwargs是一个字典，包含了训练器（trainer）的配置参数，例如训练数据、评估数据、学习率、训练轮次（epoch）等。
        trainer = Seq2SeqTrainer(**trainer_kwargs) # Seq2SeqTrainer是Hugging Face Transformers库中的一个类，专门用于序列到序列（sequence-to-sequence）的模型训练，比如T5、BART等。
        # trainer = Seq2SeqTrainer(training_args)  
        '''解释一下：训练的是T5模型，而Seq2SeqTrainer是用于训练过程的工具。'''
        
    else:
        raise ValueError
    

    trainer.train()