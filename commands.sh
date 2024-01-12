python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type llm --batch_size 64


python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type gt --batch_size 64

Standard finetuning:
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2
python standard_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2


Standard debugging:
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 5 --batch_size 2 --grad_steps 2 --max_input_length 512
python run.py --from_pretrained google/t5-v1_1-small --dataset svamp --model_type standard --label_type gt --batch_size 4

python run_code.py --from_pretrained google/t5-v1_1-small --dataset svamp --llm gt --model_type standard --eval_steps 5 --label_type gt --batch_size 1 --grad_steps 2

Distill finetuning:
python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --addi_info pred_1000
python distill_finetune.py --from_pretrained google/t5-v1_1-base --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --addi_info pred_1000

Distill debugging:
python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 5 --batch_size 2 --grad_steps 2 --max_input_length 512 --addi_info pred_1000

