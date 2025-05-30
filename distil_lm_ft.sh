#!/bin/bash
python distil_lm_ft.py \
    --from_pretrained "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "./datasets/multiclinsum_mult" \
    --eval_steps 50 \
    --max_input_length 512 \
    --max_input_length 512 \
    --batch_size_train 2 \
    --batch_size_eval 2 \
    --grad_steps 1 \
    --weight 1 \
    --alpha 0.8 \
    --model_type "standard" \
    --addi_info "standard-3K-07K" \
    --parallelize \
    --train_epochs 10 \
    --max_steps 36000 \
    --bf16
