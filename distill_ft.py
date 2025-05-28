from utils.data_utils import MultiClinSumDatasetLoader
import argparse
from utils.metrics import compute_metrics_equation
from utils.train_utils import train_and_evaluate
from transformers import AutoTokenizer


SUMMARIZE_PROMPT = 'Summarize: '
EXTRACT_PROMPT = 'Extract: '


# TODO. This function should support the mapping!
def tokenizer_func(tokenizer, examples, features, max_length):
    return tokenizer(*[examples[f] for f in features], max_length=max_length, padding="max_length", truncation=True)


def run(args):

    dataset_loader = MultiClinSumDatasetLoader(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    datasets = dataset_loader.load_from_json()

    features_to_tokenize = ["input", "output"]

    if args.model_type == "standard":

        datasets = datasets.remove_columns("rationale")

        m_func = lambda item: {
            "input": SUMMARIZE_PROMPT + item["input"],
        }

    else:
        m_func = lambda item: {
            "input": SUMMARIZE_PROMPT + item["input"],
            "expl_input": EXTRACT_PROMPT + item["input"]
        }

        features_to_tokenize += ["expl_input", "rationale"]

    datasets = datasets.map(m_func)

    tokenized_datasets = datasets.map(
        lambda e: tokenizer_func(tokenizer, e, features=features_to_tokenize, max_length=args.max_input_length),
        batched=True,
        remove_columns=features_to_tokenize
    )

    print(features_to_tokenize)
    print(datasets)
    print(tokenized_datasets["train"][:5])
    exit(0)

    train_and_evaluate(args=args, run=args.run, tokenizer=tokenizer,
                       tokenized_datasets=tokenized_datasets,
                       compute_metrics=compute_metrics_equation(tokenizer))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--max_input_length', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_eval', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--run', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix', choices=["standard", "task_prefix"])
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--addi_info', type=str, default="")
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--cos_sim', action='store_true')
    parser.add_argument('--dynamic', action='store_true')

    args = parser.parse_args()

    run(args)