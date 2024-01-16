# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

from transformers import AutoTokenizer
model_name = "t5-v1_1-base"
from_pretrained = "google/{}".format(model_name)
tokenizer = AutoTokenizer.from_pretrained(from_pretrained)


"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')
        # aux_labels: rationale_output_encodings['input_ids']
        # breakpoint()
        '''
        tokenizer.decode(expl_features['labels'][1], skip_special_tokens=True)
        '''
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        # breakpoint()
        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs) # 调用了当前类的父类（或超类）的 __init__ 方法。
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        # breakpoint()
        # decoded_output = tokenizer.decode(pred_outputs['encoder_last_hidden_state'][0], skip_special_tokens=True)
        # tokenizer.decode(inputs['expl']["labels"], skip_special_tokens=True)

        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        # breakpoint()
        # 为了
        loss = self.alpha * pred_outputs.loss + (1. - self.alpha*1000) * expl_outputs.loss
        # loss = self.alpha * pred_outputs.loss*1000 + (1. - self.alpha) * expl_outputs.loss
        
        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # breakpoint()
        # 原有的逻辑
        # pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        # if self.output_rationale:
        #     expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        # else:
        #     expl_outputs = pred_outputs # placeholder only
        # 一定让它走expl这一行的逻辑
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        # breakpoint()
        # decoded_output = tokenizer.decode(inputs['expl']['inputs'][0], skip_special_tokens=True)
        # tokenizer.decode(generated_tokens, skip_special_tokens=True)


        # 现在的逻辑
        # pred_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        # expl_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )
