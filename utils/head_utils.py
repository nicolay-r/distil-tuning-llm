import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import List, Optional, Tuple, Union
import torch.nn.init as init


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class T5WithMLPHead(T5ForConditionalGeneration):
    def __init__(self, config:T5Config):
        super(T5WithMLPHead, self).__init__(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model)
        )
        # self.mlp = nn.Sequential(nn.Linear(self.config.d_model, self.config.d_model),nn.GELU(),nn.Linear(self.config.d_model, self.config.d_model))
        # breakpoint()
        # # 先以float32精度初始化权重
        # init.xavier_uniform_(self.mlp[0].weight)
        # print("Weights after initialization in float32:", self.mlp[0].weight.data)

        # # 转换模型到bfloat16并移动到GPU
        # self.mlp = self.mlp.to(device=torch.device('cuda:0'), dtype=torch.bfloat16)
        # print("Weights after conversion to bfloat16:", self.mlp[0].weight.data)

        # .to(torch.bfloat16)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        with_head:Optional[bool] = None
    ):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #True

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # 这里好像没走
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
       

        hidden_states = encoder_outputs[0]

        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        
        if with_head:
        
           
            # # Reinitialize weights using Kaiming Initialization
            # init.kaiming_uniform_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
            # print("Reinitialized Weights:", self.mlp[0].weight.data)
            breakpoint()
            # self.mlp.to(self.first_device)
            lm_logits = self.mlp(sequence_output.float())
            
        else:
            lm_logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            # labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        breakpoint()
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )