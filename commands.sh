
Standard debugging: 
python standard_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 8 --grad_steps 2 --addi_info ds 
deepspeed standard_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type standard --max_steps 10000 --eval_steps 500 --batch_size 2 --grad_steps 1 --weight 1 --alpha 0 --addi_info distill_standard --deepspeed configs/ds_config_zero2.json

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:500 deepspeed standard_finetune.py --from_pretrained google/t5-v1_1-xl --dataset medqa_d2n --llm gt --model_type standard --eval_steps 50 --batch_size 1 --grad_steps 1 --addi_info xl --deepspeed configs/ds_config_zero2.json


Distill finetuning:
CUDA_VISIBLE_DEVICES=0 python distill_finetune.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type task_prefix --eval_steps 200 --batch_size 4 --grad_steps 2 --addi_info pred_10


CUDA_VISIBLE_DEVICES=0 deepspeed distill_finetune.py --from_pretrained google/flan-t5-small --dataset medqa_d2n --model_type task_prefix --max_steps 500 --eval_steps 2 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.5 --addi_info distill_sml --deepspeed configs/ds_config_zero2.json

deepspeed distill_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type task_prefix --max_steps 10000 --eval_steps 500 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.8 --addi_info dstl_xl_cos --deepspeed configs/ds_config_zero2.json

CUDA_VISIBLE_DEVICES=0 deepspeed distill_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type task_prefix --max_steps 1000 --eval_steps 5 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.8 --addi_info dstl_xl_28_rouge --deepspeed configs/ds_config_zero2.json


deepspeed distill_finetune.py --from_pretrained google/flan-t5-xl --dataset medqa_d2n --model_type task_prefix --max_steps 10000 --eval_steps 5 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.5 --addi_info dstl_xl --deepspeed configs/ds_config_zero2.json

deepspeed adpt_finetune.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type adapter --max_steps 10000 --eval_steps 500 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.8 --addi_info distill_adpt_l --deepspeed configs/ds_config_zero2.json

deepspeed coT_step2.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type CoT --max_steps 10000 --eval_steps 500 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.5 --addi_info CoT_xl --deepspeed configs/ds_config_zero2.json

WANDB_PROJECT=huggingface-demo TASK_NAME=MRPC deepspeed distill_finetune.py --from_pretrained google/flan-t5-small --dataset medqa_d2n --model_type task_prefix --max_steps 10 --eval_steps 5 --batch_size 1 --grad_steps 1 --weight 1 --alpha 0.5 --addi_info distill_sml --deepspeed configs/ds_config_zero2.json

deepspeed --include localhost:1 standard_finetune.py --from_pretrained google/flan-t5-xxl --dataset medqa_n2d --llm gt --model_type standard --eval_steps 5 --batch_size 1 --grad_steps 4 --addi_info ds --deepspeed configs/ds_config_zero2.json


python evaluate_summarization.py --from_pretrained google/t5-v1_1-small --dataset medqa_d2n --model_type standard --addi_info _ds --best_step 10000
python evaluate_summarization.py --from_pretrained wanglab/task-a-flan-t5-large-run-3 --dataset medqa_d2n --model_type task_prefix --addi_info wang --best_step 6000 

python eval_sum_flan.py --from_pretrained google/flan-t5-large --dataset medqa_d2n --model_type standard --addi_info _flant5_large --best_step 10000 
--deepspeed configs/ds_config_eval.json 
deepspeed zero_to_fp32.py --checkpoint_dir ./global_step10000 --output_file ./ckpt


python evaluate_summarization.py  "../conf/base.yml" "../conf/taskA.yml"     output_dir="./output/taskA/fine_tune"     do_train=False     do_eval=True
python eval_sum_medqa23.py --task taskA --fn_eval_data "./full_df.csv"




CUDA_VISIBLE_DEVICES=0 
python run_summarization.py \
    --model_name_or_path google/t5-v1_1-small \
    --do_train  False \
    --do_eval  False \
    --do_predict  True \
    --train_file standard/medqa_d2n_train.json \
    --validation_file standard/medqa_d2n_valid.json \ 
    --test_file standard/medqa_d2n_test.json \
    --text_column input \ 
    --summary_column output \ 
    --source_prefix "summarize: " \ 
    --output_dir ./tmp/tst-summarization \ 
    --overwrite_output_dir True \ 
    --per_device_train_batch_size=4 \ 
    --per_device_eval_batch_size=4 \ 
    --predict_with_generate  True \ 
    --train_adapter True \ 
    --adapter_config prefix_tuning \ 
    --overwrite_output_dir  True 

def compute_loss(self, model, inputs, return_outputs=False):
    # breakpoint()
    
    pred_outputs = model(**inputs['pred'])
    expl_outputs = model(**inputs['expl'])
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    p_mean = torch.mean(pred_outputs.encoder_last_hidden_state, dim=1)
    e_mean = torch.mean(expl_outputs.encoder_last_hidden_state, dim=1)
    loss2 = 1 -cos(p_mean, e_mean)
    loss= loss2[0]*10000
    #loss = self.alpha * pred_outputs.loss*self.weight + (1. - self.alpha) * expl_outputs.loss
    #print(loss2.shape, loss.shape)
    #breakpoint()
    return_outputs = True
    return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss
    # return loss