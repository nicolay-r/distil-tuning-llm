import sys
sys.path.append("..")  # 添加上层目录到路径中，使得 utils 模块可以被找到
from utils.data_utils import MEDQADatasetLoader
import argparse
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
# from ..utils.data_utils import MEDQADatasetLoader
from utils.metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from utils.train_utils import train_and_evaluate
import wandb
from wandb import AlertLevel

    
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, to_email):
    with open("../app_keys/k.txt",'r') as k:
        psw = k.read()
    k.close()
    from_email = 'rosaliu.567@gmail.com'
    password = psw
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


def run(args):
    #### Prepare datasets
    
    dataset_loader = MEDQADatasetLoader(args.dataset, args.model_type)

    # 加载数据
    datasets = dataset_loader.load_from_json_multi()
    
    
    # 整理数据集的label和rationale
    
    train_llm_rationales, train_llm_labels = dataset_loader.load_multi_rationale(split='train')
    valid_llm_rationales, valid_llm_labels = dataset_loader.load_multi_rationale(split='valid')

    #
    #
    # # # if args.llm is not None: # 给数据集添加labels,
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
    datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
    #
    #
    # if args.llm is not None: # 重命名rationale
    if 'rationale' in datasets['train'].column_names:
        datasets = datasets.remove_columns('rationale')
    datasets = datasets.rename_column('llm_rationale', 'rationale')
    if 'output' in datasets['train'].column_names:
        datasets = datasets.rename_column('output', 'label')

    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    # tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')

    def tokenize_function(examples):
        # This function now expects batched input, so examples['input'] and examples['output'] are lists
        rationale_list = ['Immunizations', 'Alcohol', 'Lab Results', 'Past Surgical History', 'Other',
                          'Medications or Drugs', 'Family History', 'Sex', 'Past Medical History', 'Smoking',
                          'Known Allergies', 'Diagnosis', 'Age']
        modified_entries = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [] # Add a type label to distinguish between original and rationale-based entries
        }
        for rationale_name in rationale_list:
            modified_entries[f'{rationale_name}_input_ids'] = []
            modified_entries[f'{rationale_name}_attention_mask'] = []
            modified_entries[f'{rationale_name}_labels'] = []


        for idx, dialogue in enumerate(examples['input']):
            # Original dialogue tokenization
            prompt = 'Summarize the following patient-doctor dialogue in a clinical note style. DIALOGUE:'
            original_input = tokenizer(prompt+dialogue, truncation=True, padding='max_length',max_length=args.max_input_length)
            original_output = tokenizer(examples['label'][idx], truncation=True, padding='max_length', max_length=args.max_input_length)

            # Append the original input and output
            modified_entries['input_ids'].append(original_input['input_ids'])
            modified_entries['attention_mask'].append(original_input['attention_mask'])
            modified_entries['labels'].append(original_output['input_ids'])  # Assume we use input_ids for labels

            # Process each rationale key-value pair
            rationale = examples['rationale'][idx]
            for key, value in rationale.items():
                # breakpoint()
                if value == None:
                    value = 'None'

                new_input = f"Extract the {key} information from the dialogue: {dialogue}"
                tokenized_input = tokenizer(new_input, truncation=True, padding='max_length', max_length=args.max_input_length)
                # breakpoint()
                labels = tokenizer.encode(value, add_special_tokens=False)

                # Append the rationale-based entries
                modified_entries[f'{key}_input_ids'].append(tokenized_input['input_ids'])
                modified_entries[f'{key}_attention_mask'].append(tokenized_input['attention_mask'])
                modified_entries[f'{key}_labels'].append(labels)
        # breakpoint()
        modified_entries = {key: value for key, value in modified_entries.items() if len(modified_entries[key])>0}

        return modified_entries

    print("这里mei有")
    tokenized_datasets = datasets.map(
        tokenize_function,
        remove_columns= ['input', 'label', 'llm_label', 'rationale'],
        batched=True
    )
    compute_metrics = compute_metrics_equation(tokenizer)


    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--run', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    # parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=512)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    
    
    parser.add_argument('--with_head', action='store_true')
    
    parser.add_argument('--cos_sim', action='store_true')
    
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--hierarchical', action='store_true')

    args = parser.parse_args()
    
    
    run(args)
    
    # to_email = "rosaliu.567@gmail.com"
    # send_email('模型训练开始', f'您的模型{args.addi_info}已经开始训练。', to_email)
    # try:
    #     run(args)
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型训练完成', f'您的模型{args.addi_info}已经成功训练完成。', to_email)
    # except Exception as e:
    #     print(e)
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型训练出错', f'您的模型训练时遇到问题: {e}', to_email)
       