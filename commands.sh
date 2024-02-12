
Standard debugging: 
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 8 --grad_steps 2 --addi_info ds 

deepspeed standard_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 1 --grad_steps 1 --addi_info d2n --deepspeed configs/ds_config_zero2.json
CUDA_VISIBLE_DEVICES=0 deepspeed distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 50 --batch_size 4 --grad_steps 2 --weight 50 --addi_info pred_50 --deepspeed configs/ds_config_zero2.json
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:500 deepspeed standard_finetune.py --from_pretrained google/t5-v1_1-xl --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 1 --grad_steps 1 --addi_info xl --deepspeed configs/ds_config_zero2.json


Distill finetuning:
CUDA_VISIBLE_DEVICES=0 python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 200 --batch_size 4 --grad_steps 2 --addi_info pred_10
deepspeed distill_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type task_prefix --eval_steps 200 --batch_size 1 --grad_steps 1 --weight 1000 --addi_info large_200 --deepspeed configs/ds_config_zero2.json

deepspeed distill_finetune.py --from_pretrained google/flan-t5-small --dataset medqa_d2n --model_type task_prefix --eval_steps 50 --batch_size 2 --grad_steps 1 --weight 1000 --addi_info flan --deepspeed configs/ds_config_zero2.json


python distill_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 1 --grad_steps 2 --addi_info pred_1000

Distill Inference:
python inference.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --addi_info pred_10 --best_step 10000

Sft inference:
python inference.py --from_pretrained google/t5-v1_1-large --dataset medqa_d2n --model_type standard --addi_info _d2n --best_step 10000

deepspeed --include localhost:1 standard_finetune.py --from_pretrained google/flan-t5-xxl --dataset medqa_n2d --llm gt --model_type standard --eval_steps 5 --batch_size 1 --grad_steps 4 --addi_info ds --deepspeed configs/ds_config_zero2.json



# python eval_official.py --from_pretrained google/t5-v1_1-large --dataset medqa_d2n --model_type standard --addi_info _n2d --best_step 10000


python evaluate_summarization.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type standard --addi_info _ds --best_step 10000
python evaluate_summarization.py --from_pretrained wanglab/task-a-flan-t5-large-run-3 --dataset medqa_d2n --model_type task_prefix --addi_info wang --best_step 6000 

python eval_sum_flan.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type standard --addi_info _flant5_large --best_step 10000 
--deepspeed configs/ds_config_eval.json 
deepspeed zero_to_fp32.py --checkpoint_dir ./global_step10000 --output_file ./ckpt


python evaluate_summarization.py  "../conf/base.yml" "../conf/taskA.yml"     output_dir="./output/taskA/fine_tune"     do_train=False     do_eval=True
python eval_sum_medqa23.py --task taskA --fn_eval_data "./full_df.csv"




CUDA_VISIBLE_DEVICES=0 python run_summarization.py 
    --model_name_or_path google/t5-v1_1-small 
    --do_train  True 
    --do_eval  True
    --do_predict  True
    --train_file standard/medqa_d2n_train.json 
    --validation_file standard/medqa_d2n_valid.json 
    --test_file standard/medqa_d2n_test.json
    --text_column input 
    --summary_column output 
    --source_prefix "summarize: " 
    --output_dir ./tmp/tst-summarization 
    --overwrite_output_dir True 
    --per_device_train_batch_size=4 
    --per_device_eval_batch_size=4 
    --predict_with_generate  True 
    --train_adapter True 
    --adapter_config prefix_tuning 
    --overwrite_output_dir  True