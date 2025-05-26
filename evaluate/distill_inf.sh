#!/bin/bash

MODEL="google/flan-t5-small"
TEST_FP="../datasets/multiclinsum/task_prefix/test.json"  # Provided to the script by the submission system
OUTPUT_DIR="./results/${MODEL}"
CKPT_DIR="../ckpts/task_prefix/flan-t5-xl_distill_xl_28/checkpoint-37"

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the dialogue column
# - Set the batch size to one to avoid OOM errors
# - Turn off mixed precision to avoid errors on CPUs and some GPUs
# - Set evaluation_strategy="'no'" and load_best_model_at_end=false to avoid evaluation
# - Set bertscore_model_type=null and bleurt_checkpoint=null to avoid loading them
# - Use the run=1 argument to ensure that the output file is named correctly
python3 ./run_summarization_old.py "./conf/base.yml" "./conf/taskA.yml" output_dir="$OUTPUT_DIR" \
    model_name_or_path=MODEL \
    summary_column="dialogue" \
    train_file=null \
    validation_file=null \
    test_file="$TEST_FP" \
    per_device_eval_batch_size=8 \
    fp16=false \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true \
    evaluation_strategy="'no'" \
    load_best_model_at_end=false \
    bertscore_model_type=null \
    bleurt_checkpoint=null \
    # model_type=task_prefix \
    # checkpoint_dir="$CKPT_DIR" \
    
python eval_sum_medqa23.py --task taskA --fn_eval_data $OUTPUT_DIR