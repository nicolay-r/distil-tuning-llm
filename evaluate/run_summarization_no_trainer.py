import argparse
import json
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import re
import os
from numpy import *



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a transformers model on a summarization task")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--validation_file", type=str, required=True, help="A json file containing the validation data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--max_length", type=int, default=128, help="Max length for the summaries.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the evaluation dataloader.")
    return parser.parse_args()


def postprocess_text(predictions):
    """
    Post-process the model's text predictions to clean up the output.
    """
    processed_texts = []
    for text in predictions:
        # Strip unnecessary spaces
        text = text.strip()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
         # Extract the content following 'Section text:'
        match = re.search(r'Section text: (.+)', text)
        if match:
            # Extract only the content after 'Section text:'
            text = match.group(1)
        
        # More processing can be added here

        processed_texts.append(text)
    return processed_texts

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset
    dataset = load_dataset('json', data_files={'validation': args.validation_file})
    eval_dataset = dataset['validation']

    # Prepare data loader
    def collate_fn(batch):
        inputs = [x['input'] for x in batch]
        targets = [x['output'] for x in batch]
        inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=1024, return_tensors="pt") #max_length是512或者1024都没太大差别
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return inputs, targets

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Metrics
    rouge = load_metric("rouge")
    bert_score = load_metric("bertscore")
    bleurt = load_metric("bleurt", "bleurt-base-128")

    results = []
    predictions_text = []

    for batch, targets in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids, 
                                     attention_mask=attention_mask, 
                                     max_length=args.max_length,
                                     top_p=0.9,  # Apply nucleus sampling with top-p
                                     do_sample=True,  # Enable sampling
                                     repetition_penalty=1.2,
                                     temperature=0.7,
                                     
                                     
                                    )
        
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = postprocess_text(predictions)  # Apply postprocessing here

        predictions_text.extend(predictions)
        
        for ref, pred, inp in zip(targets, predictions, eval_dataset['input']):
            results.append({
                "input": inp,
                "output": ref,
                "prediction": pred
            })
            rouge.add(prediction=pred, reference=ref)
            bert_score.add(prediction=pred, reference=ref)
            bleurt.add(prediction=pred, reference=ref)

    # Save predictions to text file
    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/generated_predictions.txt", "w") as f:
        for pred in predictions_text:
            f.write(pred + "\n")

    # Save inputs, outputs, predictions to CSV
    pd.DataFrame(results).to_csv(f"{args.output_dir}/generated_predictions_df.csv", index=False)

    # Calculate and save metrics
    rouge_result = rouge.compute()
    bert_result = bert_score.compute(lang="en")
    bleurt_result = bleurt.compute()
    final_scores = {
        "rouge1": rouge_result["rouge1"].mid.fmeasure,
        "rouge2": rouge_result["rouge2"].mid.fmeasure,
        "rougeL": rouge_result["rougeL"].mid.fmeasure,
        "rougeLsum": rouge_result["rougeLsum"].mid.fmeasure,
        "bertscore_precision": mean(bert_result["precision"]),
        "bertscore_recall": mean(bert_result["recall"]),
        "bertscore_f1": mean(bert_result["f1"]),
        "bleurt": bleurt_result["scores"][0]
    }
    with open(f"{args.output_dir}/all_results.json", "w") as f:
        json.dump(final_scores, f, indent=4)
        


if __name__ == "__main__":
    main()
    