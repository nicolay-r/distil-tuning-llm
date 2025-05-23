import sys
sys.path.append("..")  # 添加上层目录到路径中，使得 utils 模块可以被找到
from utils.data_utils import MEDMultilingual2025DatasetLoader
import argparse
from utils.metrics import compute_metrics_equation
from utils.train_utils import train_and_evaluate
from transformers import AutoTokenizer
    
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
    dataset_loader = MEDMultilingual2025DatasetLoader(args.dataset)

    # 加载数据
    datasets = dataset_loader.load_from_json_rationale()
    
    # 整理数据集的label和rationale
    train_llm_rationales, train_llm_labels = dataset_loader.load_rationale_data(split='train')
    # test_llm_rationales, test_llm_labels = dataset_loader.load_rationale_data(split='test')
    valid_llm_rationales, valid_llm_labels = dataset_loader.load_rationale_data(split='valid')

    # if args.llm is not None: # 给数据集添加labels,
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    # datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    # datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
    
    datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
    datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)

    # if args.llm is not None: # 重命名rationale
    if 'rationale' in datasets['train'].column_names:
        datasets = datasets.remove_columns('rationale')
    datasets = datasets.rename_column('llm_rationale', 'rationale')
    if 'output' in datasets['train'].column_names:
        datasets = datasets.rename_column('output', 'label')
        
    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    # tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        '''
        tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True) : (input from train set)
        'predict: Doctor: What brings you back into the clinic today, miss? 
        Patient: I came in for a refill of my blood pressure medicine. 
        Doctor: It looks like Doctor Kumar followed up with you last time regarding your hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis and kidney stones. Have you noticed any changes or do you have any concerns regarding these issues? Patient: No. Doctor: Have you had any fever or chills, cough, congestion, nausea, vomiting, chest pain, chest pressure? Patient: No. Doctor: Great. Also, for our records, how old are you and what race do you identify yourself as? Patient: I am seventy six years old and identify as a white female.'
        len(model_inputs["input_ids"]) = 1000

        '''
        model_inputs = tokenizer(['Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Dialogue:' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        expl_model_inputs = tokenizer(['Extract the key information from the dialogue, Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Dialogue: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        # breakpoint()
        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['label'], max_length=args.gen_max_len, truncation=True)
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=args.gen_max_len, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        return model_inputs

    print("这里mei有")
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
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--run', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
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