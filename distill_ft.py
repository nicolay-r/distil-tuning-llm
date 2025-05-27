from utils.data_utils import MultiClinSumDatasetLoader
import argparse
from utils.metrics import compute_metrics_equation
from utils.train_utils import train_and_evaluate
from transformers import AutoTokenizer


def wrap_and_tokenize(tokenizer, examples, provide_expl):

    model_inputs = tokenizer(
        [
            'Summarize the following patient-doctor dialogue. Include all medically relevant information, '
            'including family history, diagnosis, past medical (and surgical) history, immunizations, '
            'lab results and known allergies. Dialogue:' + text for text in examples['input']
        ],
        max_length=args.max_input_length,
        truncation=True)

    with tokenizer.as_target_tokenizer():
        label_output_encodings = tokenizer(examples['label'], max_length=args.gen_max_len, truncation=True)
        rationale_output_encodings = tokenizer(examples['rationale'], max_length=args.gen_max_len, truncation=True)

    model_inputs['labels'] = label_output_encodings['input_ids']

    if provide_expl:
        expl_model_inputs = tokenizer(
            [
                'Extract the key information from the dialogue, Include all medically relevant information, '
                'including family history, diagnosis, past medical (and surgical) history, immunizations, '
                'lab results and known allergies. Dialogue: ' + text for text in examples['input']
            ],
            max_length=args.max_input_length,
            truncation=True)

        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

    return model_inputs


def run(args):

    dataset_loader = MultiClinSumDatasetLoader(args.dataset)

    datasets = dataset_loader.load_from_json()

    for split in ['train', 'valid']:
        rationales, labels = dataset_loader.load_rationale_data(split=split)
        datasets[split] = datasets[split].add_column('llm_label', labels)
        datasets[split] = datasets[split].add_column('llm_rationale', rationales)

    if 'rationale' in datasets['train'].column_names:
        datasets = datasets.remove_columns('rationale')

    datasets = datasets.rename_column('llm_rationale', 'rationale')
    if 'output' in datasets['train'].column_names:
        datasets = datasets.rename_column('output', 'label')
        
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    tokenized_datasets = datasets.map(
        lambda examples: wrap_and_tokenize(tokenizer=tokenizer,
                                           examples=examples,
                                           provide_expl=args.model_type == "task_prefix"),
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
    parser.add_argument('--gen_max_len', type=int, default=128)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--cos_sim', action='store_true')
    parser.add_argument('--dynamic', action='store_true')

    args = parser.parse_args()

    run(args)