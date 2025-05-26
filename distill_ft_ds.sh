#!/bin/bash
# Executes fine-tuning of the Flan-T5 XL model with specialized adapter configuration using DeepSpeed

# Model, dataset, and configuration settings
MODEL="google/flan-t5-small"
DATASET="multiclinsum"
CONFIG_FILE="distill_ft_ds_zero2.json"

# Training parameters
TRAIN_EPOCHS=10
MAX_STEPS=36000
EVAL_STEPS=10
BATCH_SIZE_TRAIN=1
BATCH_SIZE_EVAL=32
GRAD_STEPS=1
WEIGHT=1
ALPHA=0.8
ADDITIONAL_INFO="Additional-Notes"

export DS_SKIP_CUDA_CHECK=1

deepspeed distill_ft.py \
    --from_pretrained $MODEL \
    --dataset $DATASET \
    --eval_steps $EVAL_STEPS \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_eval $BATCH_SIZE_EVAL \
    --grad_steps $GRAD_STEPS \
    --weight $WEIGHT \
    --alpha $ALPHA \
    --addi_info $ADDITIONAL_INFO \
    --parallelize \
    --deepspeed $CONFIG_FILE \
    --train_epochs $TRAIN_EPOCHS\
    --max_steps $MAX_STEPS \
    # --bf16 \
