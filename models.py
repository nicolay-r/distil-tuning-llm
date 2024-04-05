from transformers import T5ForConditionalGeneration, AdapterConfig

class MultiLossT5(T5ForConditionalGeneration):
    def __init__(self, t5_pretrained_model_name_or_path):
        super().__init__(self.from_pretrained(t5_pretrained_model_name_or_path))

        # 为对话摘要任务添加Adapter
        self.summary_adapter_name = 'summary'
        summary_adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16)
        self.add_adapter(self.summary_adapter_name, config=summary_adapter_config)

        # 为结构化信息生成任务添加Adapter
        self.info_adapter_name = 'info'
        info_adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16)
        self.add_adapter(self.info_adapter_name, config=info_adapter_config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        summary_labels=None,
        info_labels=None,
        return_dict=None,
    ):
        # 设置返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 结构化信息生成任务
        if info_labels is not None:
            self.set_active_adapters(self.info_adapter_name)
            info_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=info_labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
            info_loss = info_outputs.loss

        # 对话摘要任务
        if summary_labels is not None:
            self.set_active_adapters(self.summary_adapter_name)
            summary_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=summary_labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict
            )
            summary_loss = summary_outputs.loss

        # 根据是否提供标签计算总损失
        if summary_labels is not None and info_labels is not None:
            total_loss = summary_loss + info_loss
        elif summary_labels is not None:
            total_loss = summary_loss
        elif info_labels is not None:
            total_loss = info_loss
        else:
            total_loss = None

        if not return_dict:
            return total_loss, summary_outputs, info_outputs

        return {
            'loss': total_loss,
            'summary_loss': summary_loss if summary_labels is not None else None,
            'info_loss': info_loss if info_labels is not None else None,
            'summary_outputs': summary_outputs if summary_labels is not None else None,
            'info_outputs': info_outputs if info_labels is not None else None,
        }

if __name__ == '__main__':
    my_model = MultiLossT5("google/flan-t5-small")
    