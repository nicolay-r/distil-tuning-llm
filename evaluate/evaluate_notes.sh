TEST_FP="../datasets/medqa_d2n/task_prefix/medqa_d2n_test2.json"  # Provided to the script by the submission system
OUTPUT_DIR="./results/MeDistill_28_test2"
CACHE_DIR="./cache/"

python3 ./evaluate_notes.py \
        --references_fp $OUTPUT_DIR \
        --task "B" \
        --cache_dir $CACHE_DIR