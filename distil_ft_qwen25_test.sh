#!/bin/bash
python distil_lm_qwen25.py \
    --from_pretrained "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "./resources/multiclinsum_rationale_mult" \
    --save_and_eval_steps 100 \
    --max_input_length 1000 \
    --max_output_length 512 \
    --batch_size_train 1 \
    --batch_size_eval 1 \
    --eval_accumulation_steps 1 \
    --lr 1e-05 \
    --grad_steps 1 \
    --alpha 0.8 \
    --model_type "distill" \
    --description "distill-3K-07K" \
    --train_epochs 3 \
    --save_and_eval_steps 10 \
    --bf16
