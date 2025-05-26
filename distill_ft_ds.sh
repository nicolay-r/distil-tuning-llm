#!/bin/bash

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
    --from_pretrained "google/flan-t5-small" \
    --dataset "multiclinsum_en_test" \
    --eval_steps $EVAL_STEPS \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_eval $BATCH_SIZE_EVAL \
    --grad_steps $GRAD_STEPS \
    --weight $WEIGHT \
    --alpha $ALPHA \
    --addi_info $ADDITIONAL_INFO \
    --parallelize \
    --deepspeed "distill_ft_ds_zero2.json" \
    --train_epochs $TRAIN_EPOCHS\
    --max_steps $MAX_STEPS \
    # --bf16 \
