from transformers import BertModel, BertConfig
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_size, output_size):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(output_size, input_size)
    
    def forward(self, x):
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x

# 假设使用Bert模型作为基础
class BertWithAdapters(BertModel):
    def __init__(self, config):
        super(BertWithAdapters, self).__init__(config)
        # 假设插入适配器到每个encoder层
        self.adapters = nn.ModuleList([Adapter(config.hidden_size, config.hidden_size // 2) for _ in range(config.num_hidden_layers)])
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        for i, layer_output in enumerate(sequence_output):
            layer_output = self.adapters[i](layer_output)
            sequence_output[i] = layer_output
        outputs[0] = sequence_output
        return outputs

# 创建并训练模型1
config = BertConfig()
model1 = BertWithAdapters(config)

# 训练适配器...

# 将适配器迁移到模型2
model2 = BertWithAdapters(config)
model2.adapters.load_state_dict(model1.adapters.state_dict())
