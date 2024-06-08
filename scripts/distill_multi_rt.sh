#!/bin/bash
# Executes fine-tuning of the Flan-T5 XL model with specialized adapter configuration using DeepSpeed

# Model, dataset, and configuration settings
MODEL="google/flan-t5-small"
DATASET="medqa_d2n"
CONFIG_FILE="../configs/ds_config_zero2.json"
MODEL_TYPE="task_prefix"

# Training parameters
TRAIN_EPOCHS=10
MAX_STEPS=36000
EVAL_STEPS=5
BATCH_SIZE_TRAIN=1
BATCH_SIZE_EVAL=32
GRAD_STEPS=1
WEIGHT=1
ALPHA=0.8
ADDITIONAL_INFO="MeDistill_28_softlabel_ep10"


# Run the DeepSpeed training command
cd ../examples  # 确保从 examples 目录执行

deepspeed distill_multi.py \
    --from_pretrained $MODEL \
    --dataset $DATASET \
    --model_type $MODEL_TYPE \
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
    # 
    # --bf16 \
    # --dynamic
    # --peft

    # --cos_sim
    # --bf16 \
    # --with_head \

    