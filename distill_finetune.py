import argparse
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from data_utils import SVAMPDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate


def run(args):
    #### Prepare datasets
    if args.dataset == 'svamp': #设置哪个数据加载器
        dataset_loader = SVAMPDatasetLoader()
    else:
        raise ValueError
    # 加载数据
    datasets = dataset_loader.load_from_json_rationale()
    
    
    # 整理数据集的label和rationale
   
    train_llm_rationales, train_llm_labels = dataset_loader.load_rationale_data(split='train')
    test_llm_rationales, test_llm_labels = dataset_loader.load_rationale_data(split='test')
    

    if args.llm is not None: # 给数据集添加labels
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
        

    # 处理验证集
    if args.subsample < 1.0: # 切分训练集和验证集？
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if dataset_loader.has_valid: # 如果数据集里有验证集
        if args.llm is None:
            pass
        else: 
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_rationale_data(split='valid')
        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
        # breakpoint()
    else:  # 如果数据集里没有验证集
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)
        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })

    # 选择不同的计算评估的方式，如果有teacher模型的预测标签，目前数据里没有，gt: Use GT label for training  llm: Use LLM predicted label for training
    if args.label_type == 'gt': 
        pass
    elif args.label_type == 'llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'asdiv']:
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])
    else:
        raise ValueError

    if args.llm is not None: # 重命名rationale
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')
        if 'output' in datasets['train'].column_names:
            datasets = datasets.rename_column('output', 'label')
        # breakpoint()
        

    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    
    def tokenize_function(examples):
        '''
        tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True) : (input from train set)
        'predict: Doctor: What brings you back into the clinic today, miss? Patient: I came in for a refill of my blood pressure medicine. Doctor: It looks like Doctor Kumar followed up with you last time regarding your hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis and kidney stones. Have you noticed any changes or do you have any concerns regarding these issues? Patient: No. Doctor: Have you had any fever or chills, cough, congestion, nausea, vomiting, chest pain, chest pressure? Patient: No. Doctor: Great. Also, for our records, how old are you and what race do you identify yourself as? Patient: I am seventy six years old and identify as a white female.'
        len(model_inputs["input_ids"]) = 1000

        '''
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        # breakpoint()
        expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        # breakpoint()

        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        # breakpoint()
        return model_inputs


    # 不懂这是啥意思，目前猜测，是因为tokenize_function里，已经把这些都tokenize了，所以就不再保留原来的text了，只把tokenizer 传进去
    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )
   
    compute_metrics = compute_metrics_equation(tokenizer)


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
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    run(args)