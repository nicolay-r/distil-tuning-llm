conda create --name distill python=3.9 -y

conda activate distill
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* wandb accelerate==0.22.0 deepspeed==0.10.1

wandb login 7386dd9169b97829ec6b24f3587dbaf4967ca91e



export LD_LIBRARY_PATH=/home/$USER/.conda/envs/$ENVNAME/lib:/usr/local/cuda-11.6/lib64



conda create --name distill python=3.10 -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

deepspeed==0.12.3或者0.12.0
pip install chardet


pip install transformers datasets sentencepiece protobuf wandb accelerate deepspeed==0.12.3 chardet


Evaluation steps:
1. 
cd zero_to_fp32.py目录下
python zero_to_fp32.py . pytorch_model.bin

2. 
cd ./inference
sh decode_taskA_run1.sh
copy generated_predictions_df.csv to ./distill-d2n
3.
cd distill-d2n
python eval_sum_medqa23.py --task taskA --fn_eval_data "./output/9500/generated_predictions_df.csv"


	rouge1 -> 0.36
	rouge2 -> 0.169
	rougeL -> 0.305
	rougeLsum -> 0.305
	bertscore_precision -> 0.716
	bertscore_recall -> 0.678
	bertscore_f1 -> 0.691
	bleurt -> 0.515