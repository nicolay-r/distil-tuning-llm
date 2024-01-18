


python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type llm --batch_size 64


python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type gt --batch_size 64

Standard finetuning:
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 1 --grad_steps 2
python standard_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 1 --grad_steps 2
python standard_finetune.py --from_pretrained google/t5-v1_1-large --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 2 --grad_steps 2

Standard debugging:
deepspeed standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 24 --grad_steps 2 --addi_info ds --deepspeed configs/ds_config_zero2.json
deepspeed distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 8 --grad_steps 2 --alpha 0.8 --addi_info pred_8_2 --deepspeed configs/ds_config_zero2.json

python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2 

python standard_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2 --max_input_length 512
python run.py --from_pretrained google/t5-v1_1-small --dataset svamp --model_type standard --label_type gt --batch_size 4

python run_code.py --from_pretrained google/t5-v1_1-small --dataset svamp --llm gt --model_type standard --eval_steps 5 --label_type gt --batch_size 1 --grad_steps 2

Distill finetuning:
CUDA_VISIBLE_DEVICES=0 python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --addi_info pred_9_1
python distill_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 1 --grad_steps 2 --addi_info pred_1000

Distill debugging:
python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --max_input_length 512 --addi_info pred_1000
python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --alpha 0.8 --addi_info pred_8_2 --deepspeed configs/ds_config_zero2.json


Distill Inference:
python inference.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --addi_info pred_8_2 --best_step 10000

Sft inference:
python inference.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type standard --addi_info _ds --best_step 10000

deepspeed standard_finetune.py --from_pretrained google/flan-t5-xxl --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 4 --grad_steps 2 --addi_info ds --deepspeed configs/ds_config_zero2.json
