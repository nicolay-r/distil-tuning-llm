#!/bin/bash
# Executes fine-tuning of the Flan-T5 large model with specified parameters using DeepSpeed

# Model, dataset, and configuration settings
MODEL="google/flan-t5-small"
DATASET="medqa_d2n"
CONFIG_FILE="../configs/ds_config_zero2.json"
MODEL_TYPE="standard"

# Training parameters
MAX_STEPS=10000
EVAL_STEPS=5
BATCH_SIZE_TRAIN=2
BATCH_SIZE_EVAL=4
GRAD_STEPS=1
WEIGHT=1
ALPHA=0
ADDITIONAL_INFO="sft_ft"


# Run the DeepSpeed training command
cd ../examples  # 确保从 examples 目录执行

deepspeed standard_finetune.py \
    --from_pretrained $MODEL \
    --dataset $DATASET \
    --model_type $MODEL_TYPE \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_eval $BATCH_SIZE_EVAL \
    --grad_steps $GRAD_STEPS \
    --weight $WEIGHT \
    --alpha $ALPHA \
    --addi_info $ADDITIONAL_INFO \
    --deepspeed $CONFIG_FILE \
    --bf16 \
    --parallelize

