import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn import CrossEntropyLoss
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class T5WithMLPHead(nn.Module):
    def __init__(self, expl_output, model, mlp_hidden_dim):
        super(T5WithMLPHead, self).__init__()
        self.expl_output = expl_output
        self.t5 = model  # 直接使用传递的T5模型实例

        self.mlp = nn.Sequential(
            nn.Linear(self.t5.model_dim, mlp_hidden_dim),  # d_model是T5模型的隐藏层维度
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 32318),
        )

    def forward(self, inputs, attention_mask=None):
        
        # input_ids = inputs['input_ids'].to(device)
        labels = inputs['labels']
        # attention_mask = inputs['attention_mask'].to(device)
        # decoder_input_ids = inputs['decoder_input_ids'].to(device)
        # outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, return_dict=True)
        # 使用encoder的最后一个隐藏状态
        last_hidden_states = self.expl_output.encoder_last_hidden_state

        # 应用MLP
        mlp_output = self.mlp(last_hidden_states[:, 0, :].float()) # 取encoder最后层的CLS token的输出 
        labels = torch.argmax(labels, dim=1)  # 将独热编码转换为类索引
        labels = labels.long()  # 确保labels是长整型

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(mlp_output, labels)
        
        return mlp_output, loss 