#!/bin/bash
python distil_ft_qwen25.py \
    --from_pretrained "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "./resources/multiclinsum_rationale_mult" \
    --max_input_length 3078 \
    --max_output_length 762 \
    --batch_size_train 2 \
    --batch_size_eval 1 \
    --lr 1e-05 \
    --grad_steps 1 \
    --alpha 0.8 \
    --model_type "standard" \
    --description "standard-3K-07K" \
    --hub_model_id "nicolay-r/qwen25-05b-multiclinsum-standard" \
    --train_epochs 3 \
    --save_and_eval_steps 250 \
    --bf16
