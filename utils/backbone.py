import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration


class DistillBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        # self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
         # 假设T5模型输出的最后一个维度是隐藏层的大小
        self.task_head = nn.Linear(self.engine.config.d_model, num_labels)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, **kwargs):
        
        
        outputs = self.engine(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask)
        
        # 取得decoder的最后一个输出作为分类任务的输入
        last_hidden_states = outputs.last_hidden_state[:, -1, :]
        logits = self.task_head(last_hidden_states)
        return self.activation(logits)
        

    # def generate(self, **kwargs):
    #     input_ids, input_masks = [kwargs[w] for w in '\
    #     input_ids, input_masks'.strip().split(', ')]
    #     output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
    #                                   max_length=self.config.max_length)
    #     dec = [self.tokenizer.decode(ids) for ids in output]
    #     output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
    #     return output

    # def evaluate(self, **kwargs):
    #     input_ids, input_masks = [kwargs[w] for w in '\
    #     input_ids, input_masks'.strip().split(', ')]
    #     output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
    #     dec = [self.tokenizer.decode(ids) for ids in output]
    #     label_dict = {w: i for i, w in enumerate(self.config.label_list)}
    #     output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
    #     return output