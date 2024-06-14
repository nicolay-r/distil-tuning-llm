#!/bin/bash
# Submits our Flan-T5 large based approach for task A

TEST_FP="../datasets/medqa_d2n/task_prefix/medqa_d2n_test2.json"  # Provided to the script by the submission system

OUTPUT_DIR="./results/MeDistill_wang2_t5"
# /root/distill-d2n/ckpts/task_prefix/flan-t5-large_dstl_xl/checkpoint-250/pytorch_model.bin
# CKPT_DIR="../ckpts/task_prefix/flan-t5-xl_distill_xl_28/checkpoint-37"
# /root/distill-d2n/ckpts/task_prefix/flan-t5-xl_distill_xl_28/checkpoint-30
# /root/distill-d2n/ckpts/task_prefix/flan-t5-large_distill_large_28/checkpoint-10000/special_tokens_map.json
# /root/distill-d2n/ckpts/task_prefix/flan-t5-large_distill_large/checkpoint-9000/pytorch_model.bin

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the dialogue column
# - Set the batch size to one to avoid OOM errors
# - Turn off mixed precision to avoid errors on CPUs and some GPUs
# - Set evaluation_strategy="'no'" and load_best_model_at_end=false to avoid evaluation
# - Set bertscore_model_type=null and bleurt_checkpoint=null to avoid loading them
# - Use the run=1 argument to ensure that the output file is named correctly
# 
python3 ./run_summarization_no_trainer.py \
    --model_name_or_path "wanglab/task-a-flan-t5-large-run-2" \
    --validation_file "$TEST_FP" \
    --max_length 1024 \
    --batch_size 2 \
    --output_dir "$OUTPUT_DIR" \
    # model_type=task_prefix \
    # checkpoint_dir="$CKPT_DIR" \
    
# python eval_sum_medqa23.py --task taskA --fn_eval_data $OUTPUT_DIR


