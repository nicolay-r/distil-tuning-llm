

import argparse

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from dataset_xx import SVAMPDatasetLoader


from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate


def run(args):
    #### Prepare datasets
    if args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    else:
        raise ValueError

    if args.dataset == 'asdiv':
        print('''args.dataset == asdiv''')
    else:
        datasets = dataset_loader.load_from_json()
    # breakpoint()
    
    if args.llm == 'gt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gt_preds(split='test')
    elif args.llm == 'palm':
        train_llm_rationales, train_llm_labels = dataset_loader.load_rationale_data(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_rationale_data(split='test')
    else:
        raise ValueError

    if args.llm is not None:
        # breakpoint()
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        print(len(datasets['train']))
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
        print(len(datasets['test']))
        # breakpoint()
    # 给验证集添加 label
    if args.subsample < 1.0: 
        # 切分train和valid集
        print("小于")
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if dataset_loader.has_valid:
        print("有")
        pass
    else:
        print('没有')
        
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })
    
    if args.label_type == 'gt': # groundtruth
        print("是gt")
        pass

    else:
        raise ValueError
    breakpoint()
    # 不太重要的地方，整理列名
    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            print("这里有")
            datasets = datasets.remove_columns('rationale')
        print("这里没有")
        datasets['train'].column_names
        datasets = datasets.rename_column('llm_rationale', 'rationale')
    # breakpoint()
    


    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    
    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            # breakpoint()
            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            # if task_type == "d2n":
            #     max_length = 
            # 使用 tokenizer 将 examples 中的 'input' 字段的文本进行分词处理。
            # 设置最大长度为 args.max_input_length，并在超出时截断文本。 
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=1024, truncation=True) #设置最大长度为 256，并在超出时截断文本。

            model_inputs['labels'] = label_output_encodings['input_ids']
            
            return model_inputs

    else:
        raise ValueError


    if args.llm is None:
        print("这里有")
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        print("这里mei有")
        #走了这里
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            
            batched=True
        )
        # breakpoint()
    

    if args.model_type == 'standard':
        print("走这里standard")
        if args.dataset not in ['svamp', 'asdiv']:
            print("text")
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            print("equation")
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        print("走这里task_prefix")
        if args.dataset not in ['svamp', 'asdiv']:
            print("走这里")
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            print("没走这里")
            compute_metrics = compute_metrics_equation(tokenizer)
            
    print(len(datasets['valid']))
    
    
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    # parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--llm', type=str)

    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--task_type', type=str, default='d2n')

    args = parser.parse_args()

    run(args)