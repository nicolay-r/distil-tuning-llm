#!/bin/bash
python distill_ft.py \
    --from_pretrained "google/flan-t5-small" \
    --dataset "multiclinsum_en_test" \
    --eval_steps 10 \
    --batch_size_train 1 \
    --batch_size_eval 32 \
    --grad_steps 1 \
    --weight 1 \
    --alpha 0.8 \
    --addi_info "Additional-Notes" \
    --parallelize \
    --train_epochs 10 \
    --max_steps 36000 \
    # --bf16 \
