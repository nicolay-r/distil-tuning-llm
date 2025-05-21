from transformers import T5ForConditionalGeneration, T5Attention

class CustomT5Attention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        # 可以在这里添加自定义的初始化代码，例如不同的参数化形式

    def forward(self, *args, **kwargs):
        # 在这里实现自定义的attention逻辑
        # 可以完全重写attention，或者在调用super().forward之前后添加额外的逻辑
        return super().forward(*args, **kwargs)

class ModifiedT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 替换原有的attention层
        self.encoder.block[0].layer[0].SelfAttention = CustomT5Attention(config)
        self.decoder.block[0].layer[0].SelfAttention = CustomT5Attention(config, has_relative_attention_bias=True)

