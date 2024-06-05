#!/bin/bash
# Submits our Flan-T5 large based approach for task A

TEST_FP="../datasets/medqa_d2n/task_prefix/medqa_d2n_test2.json"  # Provided to the script by the submission system

OUTPUT_DIR="./results/xiao_28"
# /root/distill-d2n/ckpts/task_prefix/flan-t5-large_dstl_xl/checkpoint-250/pytorch_model.bin
# CKPT_DIR="../ckpts/task_prefix/flan-t5-large_MeDistill_28_rougeAve/checkpoint-7800/"

# ../ckpts/task_prefix/flan-t5-large_MeDistill_28_rougeAve/checkpoint-7200/pytorch_model.bin


python3 ./run_summarization_old.py "./conf/base.yml" "./conf/taskA.yml" output_dir="$OUTPUT_DIR" \
    model_name_or_path="Xiaolihai/flan-t5-large_MeDistill_28" \
    summary_column="dialogue" \
    train_file=null \
    validation_file=null \
    test_file="$TEST_FP" \
    per_device_eval_batch_size=32 \
    fp16=false \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true \
    eval_strategy="'no'" \
    load_best_model_at_end=false \
    bertscore_model_type=null \
    bleurt_checkpoint=null \
    # model_type=task_prefix \
    # checkpoint_dir="$CKPT_DIR" \
     
ls 
python3 ./eval_sum_medqa23.py \
    --fn_eval_data $OUTPUT_DIR \

exit

