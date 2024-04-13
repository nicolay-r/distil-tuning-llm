import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn import CrossEntropyLoss


class T5WithMLPHead(nn.Module):
    def __init__(self, model, mlp_hidden_dim, output_dim, device):
        super(T5WithMLPHead, self).__init__()
        self.device = device
        # self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5 = model  # 直接使用传递的T5模型实例
        # breakpoint()
        self.mlp = nn.Sequential(
            nn.Linear(self.t5.model_dim, mlp_hidden_dim),  # d_model是T5模型的隐藏层维度
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        ).to(self.device)

    def forward(self, inputs, attention_mask=None):
        # breakpoint()
        input_ids = inputs['input_ids'].to(self.device)
        labels = inputs['labels'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        decoder_input_ids = inputs['decoder_input_ids'].to(self.device)
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, return_dict=True)
        
        # 使用encoder的最后一个隐藏状态
        last_hidden_states = outputs.encoder_last_hidden_state
        # breakpoint()
        # 应用MLP
        mlp_output = self.mlp(last_hidden_states[:, 0, :].float())  # 取encoder最后层的CLS token的输出
        
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)
        # lm_logits = mlp_output
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            # breakpoint()
            # labels = labels.to(mlp_output.device)
            # breakpoint()
            loss = loss_fct(mlp_output.view(-1, mlp_output.size(-1)), labels.view(-1))
        breakpoint()

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return loss 