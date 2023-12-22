python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type llm --batch_size 64


python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type gt --batch_size 64

Standard finetuning:
python run.py --from_pretrained google/t5-v1_1-small --dataset svamp --model_type standard --label_type gt --batch_size 4
python run_code.py --from_pretrained google/t5-v1_1-small --dataset svamp --model_type standard --label_type gt --batch_size 4

python run_code.py --from_pretrained google/t5-v1_1-large --dataset svamp --llm gt --model_type standard --eval_steps 500 --label_type gt --batch_size 1 --grad_steps 2

Distill finetuning:
python distill_finetune.py --from_pretrained google/t5-v1_1-base --dataset svamp --llm palm --model_type task_prefix --eval_steps 50 --label_type gt --batch_size 1 --grad_steps 2

Distill debugging:
python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset svamp --model_type task_prefix --eval_steps 5 --label_type gt --batch_size 2 --grad_steps 2 --max_input_length 512


conda create --name distillsds python=3.10.6 -y
