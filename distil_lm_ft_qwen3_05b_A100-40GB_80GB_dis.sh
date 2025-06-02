#!/bin/bash
python distil_lm_ft.py \
    --from_pretrained "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "./resources/multiclinsum_rationale_mult" \
    --save_and_eval_steps 100 \
    --max_input_length 3078 \
    --max_output_length 762 \
    --batch_size_train 1 \
    --batch_size_eval 1 \
    --eval_accumulation_steps 1 \
    --lr 1e-05 \
    --grad_steps 1 \
    --alpha 0.8 \
    --model_type "distill" \
    --description "distill-3K-07K" \
    --hub_model_id "nicolay-r/qwen25-05b-multiclinsum-distil" \
    --train_epochs 3 \
    --save_and_eval_steps 250 \
    --bf16
