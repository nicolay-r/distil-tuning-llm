import sys
sys.path.append("..")  # 添加上层目录到路径中，使得 utils 模块可以被找到
import argparse
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from utils.data_utils import MEDQADatasetLoader
from utils.metrics import compute_metrics_equation_aux, compute_metrics_equation
from utils.train_utils import train_and_evaluate
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
    # breakpoint()
    #### Prepare datasets
    dataset_loader = MEDQADatasetLoader(args.dataset, args.model_type)
    
    # 加载数据
    datasets = dataset_loader.load_from_json()
    # breakpoint()
    # # 整理数据集的label

    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    
    def tokenize_function(examples):
        # 使用 tokenizer 将 examples 中的 'input' 字段的文本进行分词处理。
        # 设置最大长度为 args.max_input_length，并在超出时截断文本。
        model_inputs = tokenizer(['Summarize the following patient-doctor dialogue and Extract the key information from the dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Dialogue:' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
#         model_inputs = tokenizer(
#             examples['input'],
#             max_length=args.max_input_length,
#             truncation=True
#         )

        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['output'], max_length=args.gen_max_len, truncation=True, padding='max_length') #设置最大长度为1024，并在超出时截断文本。

        model_inputs['labels'] = label_output_encodings['input_ids']
        
        return model_inputs

    

    
    tokenized_datasets = datasets.map(
        tokenize_function,
        remove_columns=['input', 'output'],
        batched=True
    )
    

    compute_metrics = compute_metrics_equation(tokenizer)
    # compute_metrics = compute_metrics_equation(tokenizer)

    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    # parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--llm', type=str)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=1024)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--task_type', type=str, default='d2n')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)

    args = parser.parse_args()

    run(args)
    
    # to_email = "rosaliu.567@gmail.com"
    # send_email('模型训练开始', '您的模型已经开始训练。', to_email)
    # try:  
    #     run(args)
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型训练完成', '您的模型已经成功训练完成。', to_email)
    # except Exception as e:
    #     print(e)
    #     # to_email = "rosaliu.567@gmail.com"
    #     send_email('模型训练出错', f'您的模型训练时遇到问题: {e}', to_email)  
       
