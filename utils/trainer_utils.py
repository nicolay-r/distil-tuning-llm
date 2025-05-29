# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0为法国

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq, Trainer, DataCollatorForLanguageModeling
import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TaskPrefixDataCollator(DataCollatorForLanguageModeling):

    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={
                'aux_labels': 'labels',
                'expl_input_ids': 'input_ids',
                'expl_attention_mask': 'attention_mask'
            }).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        
        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Trainer):

    def __init__(self, alpha, data_collator=None,**kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.data_collator = data_collator if data_collator is not None else DataCollatorForSeq2Seq()
    
    def compute_loss(self, model, inputs, return_outputs=False):

        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        # TODO. Refactor this into single method.
        pred_labels = inputs['pred']['labels']  # Assuming true labels are here
        pred_preds = torch.argmax(pred_outputs.logits, dim=-1)
        pred_accuracy = (pred_preds == pred_labels).float().mean()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # TODO. Refactor this into single method.
        expl_labels = inputs['expl']['labels']  # Assuming true labels for explanations
        expl_preds = torch.argmax(expl_outputs.logits, dim=-1)
        expl_accuracy = (expl_preds == expl_labels).float().mean()

        wandb.log({
                'train/loss': loss,
                'train/loss_pred': pred_outputs.loss,
                'train/loss_expl': expl_outputs.loss,
                'train/pred_accuracy': pred_accuracy.item(),  # Logging prediction accuracy
                'train/expl_accuracy': expl_accuracy.item(),  # Logging explanation accuracy (if applicable)
                'learning_rate': current_lr,
                },
                step=self.state.global_step)

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        # torch.cuda.empty_cache() # 也许可以试一下
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        # breakpoint()
        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        wandb.log(
            {
                'eval/loss': loss,
                'eval/loss_pred': pred_outputs[0],
                'eval/loss_expl': expl_outputs[0]
            },
            step=self.state.global_step)

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )
