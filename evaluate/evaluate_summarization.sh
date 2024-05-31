#!/bin/bash
# Requested resources
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100l:1
# Wall time and job details
#SBATCH --time=1:00:00
#SBATCH --job-name=eval
#SBATCH --account=def-wanglab-ab
# Emails me when job starts, ends or fails
#SBATCH --mail-user=johnmgiorgi@gmail.com
#SBATCH --mail-type=FAIL
# Use this command to run the same job interactively
# salloc --mem=16G --cpus-per-task=1 --gres=gpu:v100l:1 --time=3:00:00 --account=def-wanglab-ab

### Example usage ###
# sbatch "./scripts/eval.sh" "data/TaskB-ValidationSet-SubmissionFormat.csv" "data/predictions/taskB_run1.csv" "taskB" "run1"

### Usage notes ###
# ...

### Environment ###
# module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
# module load python/3.10 StdEnv/2020 gcc/9.3.0 arrow/10.0
# PROJECT_NAME="mediqa-eval"
# ACCOUNT_NAME="rrg-wanglab"
# source "$HOME/$PROJECT_NAME/bin/activate"
# cd "$HOME/projects/$ACCOUNT_NAME/$USER/MEDIQA-Chat-2023" || exit

### Script arguments ###
# Required arguments
# FN_GOLD="$1"    # The path on disk to the JSON config file
# FN_SYS="$2"     # The path on disk to save the output to
# TASK="$A"       # Task, should be taskA or taskB
# OUTPUT_FN="$4"  # Output filename

# # Set the note column
# if [[ "$TASK" == "taskA" ]]; then
#   NOTE_COLUMN="SystemOutput2"
# else
#   NOTE_COLUMN="SystemOutput"
# fi
OUTPUT_DIR="./results/wang2_test2_split"

### Job ###
python ./evaluate_summarization.py \
    --fn_eval_data $OUTPUT_DIR \

exit