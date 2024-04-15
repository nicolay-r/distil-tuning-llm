#!/usr/bin/env python
# coding=utf-8

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from filelock import FileLock
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

# Constants for the MEDIQA-Chat @ ACL-ClinicalNLP challenge task
# A list of valid task names for the MEDIQA-Chat @ ACL-ClinicalNLP

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_type: str = field(
        default=None,
        metadata={"help": "adapter or task_prefix"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    source_suffix: Optional[str] = field(default="", metadata={"help": "A suffix to add after every source text."})


    task: str = field(
        default="A", metadata={"help": "Which challenge task to train or evaluate on. Should be one of A, B, or C."}
    )
    run: str = field(
        default="1", metadata={"help": "Which challenge run to produce predictions for. Should be one of 1, 2, or 3."}
    )
    bertscore_model_type: str = field(
        default="microsoft/deberta-large-mnli",
        metadata={
            "help": (
                "Which model to use for BERTScore. This is a good default that balances metric performance and model"
                " size. For options, see:"
                " https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing"
            )
        },
    )
    bleurt_checkpoint: str = field(
        default="BLEURT-20-D12",
        metadata={
            "help": (
                "Which checkpoint to use for BLEURT. This is a good default that balances metric performance and"
                " model size. For options, see:"
                " https://github.com/google-research/bleurt/blob/master/checkpoints.md"
            )
        },
    )

    def __post_init__(self):
        if (
            # self.dataset_name is None
            self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def parse_omega_conf() -> Dict[str, Any]:
    """A helper function to parse the command line arguments and YAML config files.

    This allows you to call the script with one or more YAML config files and CLI args.

    Example:

    python train.py config1.yml config2.yml --learning_rate=0.0001 --train_batch_size=32

    Arguments in config1.yml will overridden arguments in config2.yml, which will be overridden by any CLI args.
    """
    # Assume anything that ends in ".yml" is a YAML file to be parsed, and everything else is cli args
    cli_args = [arg for arg in sys.argv[1:][:] if not arg.endswith(".yml")]
    yml_confs = [OmegaConf.load(arg) for arg in sys.argv[1:][:] if arg.endswith(".yml")]
    cli_conf = OmegaConf.from_dotlist(cli_args)
    # Merge the YAML configs in the order they were given, with the cli args taking precedence
    conf = OmegaConf.merge(*yml_confs, cli_conf)
    # HuggingFace expects a vanilla python dict, so perform the conversion here
    return OmegaConf.to_object(conf)


def get_global_attention_mask(input_ids: List[List[int]], token_ids: List[int]) -> List[List[int]]:
    """Returns a corresponding global attention mask for `input_ids`, which is 1 for any tokens in
    `token_ids` (indicating the model should attend to those tokens) and 0 elsewhere (indicating the
    model should not attend to those tokens).

    # Parameters

    input_ids : `List[List[str]]`
        The input ids that will be provided to a model during the forward pass.
    token_ids : `List[List[str]]`
        The token ids that should be globally attended to.
    """
    # TODO (John): Ideally this would be vectorized
    global_attention_mask = [[1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids]
    return global_attention_mask


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    print("*********第一部分 开始**********")
    conf = parse_omega_conf()
    model_args, data_args, training_args = parser.parse_dict(conf)

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "omega_conf.json").write_text(json.dumps(conf, indent=2))
    print("*********第一部分 结束**********")
    # breakpoint()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        print("走了这里：training_args.should_log")
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    # breakpoint()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    data_files = {}

    data_files["test"] = data_args.test_file
    extension = data_args.test_file.split(".")[-1]

    data_files["test"] = data_args.test_file
    extension = 'json'
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_type == 'adapter':
        # from peft import PeftModel, PeftConfig
        # model_dir = model_args.checkpoint_dir
        # config = PeftConfig.from_pretrained(model_dir)
        # model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        # model = PeftModel.from_pretrained(model, model_dir)
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(model_args.checkpoint_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.checkpoint_dir)


        # model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        # model = get_peft_model(model, peft_config)
        # model_dir = model_args.checkpoint_dir
        # checkpoint = torch.load(f"{model_dir}pytorch_model.bin", map_location="cpu") #读取本地训练好的chekpoint
        # model.load_state_dict(checkpoint)
        # model.print_trainable_parameters()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_dir = model_args.checkpoint_dir
        checkpoint = torch.load(f"{model_dir}pytorch_model.bin", map_location="cpu") #读取本地训练好的chekpoint
        model.load_state_dict(checkpoint)
    
   

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    suffix = data_args.source_suffix if data_args.source_suffix is not None else ""
    suffix = "SUMMARY: "

    
    column_names = raw_datasets["test"].column_names

    text_column = "input"
    summary_column = "output"

    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs_org, targets, inputs = [], [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs_org.append(examples[text_column][i])
  
            
        inputs = [f"{prefix} {inp} {suffix}".strip() for inp in inputs_org]
        targets = examples["output"]
        # breakpoint()

        # Do some basic cleanup on the source and target texts
        inputs = [sanitize_text(inp) for inp in inputs]
        targets = [sanitize_text(tgt) for tgt in targets]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Add global attention to start tokens (for models that use it)
        model_inputs["global_attention_mask"] = get_global_attention_mask(model_inputs["input_ids"], token_ids=[0])
        # breakpoint()
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    # breakpoint()

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # breakpoint()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # breakpoint()
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                res_predictions = predict_results.predictions
                res_predictions = np.where(res_predictions != -100, res_predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    res_predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
              
                # Predictions may contain "\n", which causes this file to contain an incorrect number of lines.
                # Pointed this out to HF maintainers here: https://github.com/huggingface/transformers/issues/18992
                # but they weren't interested in fixing it.
                predictions = [sanitize_text(pred) for pred in predictions]
                pattern = r'Section text: (.*?)(?=(\', \'Section header:|$))'
                
                pred_result = []
                # breakpoint()
                i = 0
                for p in predictions:
                    i+=1
                    try:
                        matches = re.findall(pattern, p, re.DOTALL)[0][0]
                    except:
                        matches = p
                        # print("这个错了：",p)
                        # print(i)
                    pred_result.append(matches)
                # SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
                full_df = pd.DataFrame(raw_datasets['test'], columns=raw_datasets['test'].features)
                
                # breakpoint()
                full_df['prediction'] = pred_result
                full_df.to_csv
                csv_path = f'{training_args.output_dir}/generated_predictions_df.csv'
                full_df.to_csv(csv_path)
                # breakpoint()
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(pred_result))



import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, to_email):
    from_email = 'rosaliu.567@gmail.com'
    password = 'jdrb ueoq ixik tuoa'
    # jdrb ueoq ixik tuoa

    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    body = MIMEText(message, 'plain')
    msg.attach(body)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()
    
if __name__ == "__main__":
    main()
    # to_email = "rosaliu.567@gmail.com"
    # try:  
    #     main()
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型评估完成', '您的模型评估已完成。', to_email)
    # except Exception as e:
    #     print(e)
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型评估出错', f'您的模型评估时遇到问题: {e}', to_email)  
       
