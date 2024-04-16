import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torch.nn import CrossEntropyLoss
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class T5WithMLPHead(T5ForConditionalGeneration):
    def __init__(self, config:T5Config):
        super(T5WithMLPHead, self).__init__(config)
        self.mlp_hidden_dim = 1024
       
        self.last_layer =  self.lm_head
        self.mlp = nn.Sequential(
            nn.Linear(self.config.vocab_size, self.mlp_hidden_dim), # d_model是T5模型的隐藏层维度
            nn.GELU(), # GELU
            nn.Linear(self.mlp_hidden_dim, self.config.vocab_size),
        ).to(torch.bfloat16)
        
    def forward(self, input_ids, with_head=False):
        outputs = super().forward(**input_ids)
        # breakpoint()
        if with_head:
            
            labels = input_ids['labels']
            
            # 使用encoder的最后一个隐藏状态
            last_hidden_states = outputs.logits
            # breakpoint()
            # 应用MLP
            # mlp_output = self.mlp(last_hidden_states[:, 0, :].to(torch.bfloat16))
            mlp_output = self.mlp(last_hidden_states.to(torch.bfloat16))
            # labels = labels.to(mlp_output.device)
            # (Pdb) lm_logits.shape
            # torch.Size([1, 14, 32128])
            # (Pdb) labels.shape
            # torch.Size([1, 14])
            # mlp_output = self.mlp(last_hidden_states.to(torch.bfloat16))# 取encoder最后层的CLS token的输出 
            # labels = torch.argmax(labels, dim=1)  # 将独热编码转换为类索引
            # labels = labels.long()  # 确保labels是长整型

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # loss = loss_fct(mlp_output, labels)
                breakpoint()
                loss = loss_fct(mlp_output.view(-1, mlp_output.size(-1)), labels.view(-1))
            breakpoint()
            outputs['loss'] = loss
            outputs['logits'] = mlp_output
        else:
    #         tensor(5.6875, device='cuda:0', dtype=torch.bfloat16,
    #    grad_fn=<NllLossBackward0>)
            pass
        # breakpoint()
        return outputs 