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

from adapters import Seq2SeqAdapterTrainer
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from utils.data_utils import MEDQADatasetLoader
import wandb
from transformers import AutoTokenizer
# model_name = "t5-v1_1-base"
# from_pretrained = "google/{}".format(model_name)
# tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""T5 Multi-Task by Task Prefix
""" 
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')
        # breakpoint()
        '''
        tokenizer.decode(expl_features['labels'][1], skip_special_tokens=True)
        '''
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        
        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, weight, data_collator=None,**kwargs):
        super().__init__(**kwargs) # 调用了当前类的父类（或超类）的 __init__ 方法。
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.weight = weight
        self.data_collator = data_collator if data_collator is not None else DataCollatorForSeq2Seq()

    def _cosine_loss(self, pred_outputs, expl_outputs):
            # breakpoint()
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            p_mean = torch.mean(pred_outputs, dim=1)
            e_mean = torch.mean(expl_outputs, dim=1)
            cos_loss = 1 - cos(p_mean, e_mean)
            return cos_loss[0]
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        
        '''
        Seq2SeqLMOutput,  model.forward()返回内容说明：

        loss: 损失值，表示模型预测与实际标签之间的差距。在训练过程中用来优化模型参数。
        logits: 模型的原始输出，未经过Softmax转换的概率分布。它们通常用于计算损失函数和进行预测分析。
        past_key_values: 解码器的过去键值对，用于实现序列到序列模型的增量解码。它允许模型在生成新词时重用以前的计算结果，从而提高效率。
        decoder_hidden_states: 解码器的隐藏状态，包含了在解码过程中各个时间步的隐藏层状态。这些状态可以用于分析模型的内部表征。
        decoder_attentions: 解码器的注意力权重，展示了解码过程中各个输出对输入序列的关注程度。这有助于理解模型的决策过程。
        cross_attentions: 交叉注意力权重，展示了解码器如何在生成每个词时关注编码器的输出。这同样有助于分析模型的工作机制。
        encoder_last_hidden_state: 编码器的最后一个隐藏状态，可以被用作序列的表示，有时用于初始化解码器的隐藏状态。
        encoder_hidden_states: 编码器的隐藏状态，包含了编码过程中各个时间步的隐藏层状态。这些状态可以用于进一步的特征提取和分析。
        encoder_attentions: 编码器的注意力权重，展示了编码过程中各个输入之间的关联强度。
        '''
        # breakpoint()
        # 原来的
        # loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss
        # cosine-similarity之后的
        cos_loss  = self._cosine_loss(pred_outputs.encoder_last_hidden_state, expl_outputs.encoder_last_hidden_state) 
        loss = 0.6 * pred_outputs.loss + 0.2 * expl_outputs.loss + cos_loss * 0.2
        # breakpoint()
        
        # ********************Accuracy**************************
        # Example accuracy calculation for predictions
        pred_labels = inputs['pred']['labels']  # Assuming true labels are here
        pred_preds = torch.argmax(pred_outputs.logits, dim=-1)
        pred_accuracy = (pred_preds == pred_labels).float().mean()
        
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Example accuracy calculation fprint(pred_outputs[0].shape)or explanations (if applicable)
        expl_labels = inputs['expl']['labels']  # Assuming true labels for explanations
        expl_preds = torch.argmax(expl_outputs.logits, dim=-1)
        expl_accuracy = (expl_preds == expl_labels).float().mean()
        wandb.log({'train/loss': loss, 
                   'train/loss_pred': pred_outputs.loss, 
                   'train/cos_loss': cos_loss,
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
        # 原来的      
        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        
        wandb.log({'eval/loss': loss, 
                   'eval/loss_pred': pred_outputs[0], 
                   'eval/loss_expl': expl_outputs[0]                  
                   },
                  step=self.state.global_step)
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )



class CoTDataCollator(DataCollatorForSeq2Seq):
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
        
        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class CoTTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, weight, data_collator=None,**kwargs):
        super().__init__(**kwargs) # 调用了当前类的父类（或超类）的 __init__ 方法。
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.weight = weight
        self.data_collator = data_collator if data_collator is not None else DataCollatorForSeq2Seq()


    
    def _cot_steps(self, model, inputs):
        kw_output = model.generate(**inputs['pred'], max_new_tokens=1024)[0]
        # breakpoint()
        input_2 = tokenizer.decode(inputs['expl']['input_ids'][0], skip_special_tokens=True)
        
        kw_input_ids = tokenizer.batch_encode_plus([f'{input_2}{kw_output}\nSUMMARY\n'],padding=True, return_tensors='pt',truncation=True,max_length= 1024).to(device)
        # x = tokenizer.batch_encode_plus([f'{kw_input[0]}{kw_output}{kw_input[1]}'], max_length=1024, padding=True, return_tensors='pt')
        
        kw_dict = {
            'input_ids':kw_input_ids["input_ids"],
            'attention_mask': kw_input_ids['attention_mask'],
            'labels': inputs['expl']['labels'],
            'decoder_input_ids': inputs['expl']['decoder_input_ids'],
        }
        return kw_dict
    
    def compute_loss(self, model, inputs, return_outputs=False):
        '''better set batch_size = 1'''
        kw_dict = self._cot_steps(model, inputs)
        output = model(**kw_dict)
        loss = output.loss
        return loss



    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        kw_dict = self._cot_steps(model, inputs)
        output = model(**kw_dict)
        loss = output.loss
        
        return (loss, None, None)


class AdapterDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')
        # breakpoint()
        '''
        tokenizer.decode(expl_features['labels'][1], skip_special_tokens=True)
        '''
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        
        return {
            'pred': pred_features,
            'expl': expl_features,
        }
        
class AdptTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, weight, data_collator=None,**kwargs):
        super().__init__(**kwargs) # 调用了当前类的父类（或超类）的 __init__ 方法。
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.weight = weight
        self.data_collator = data_collator if data_collator is not None else DataCollatorForSeq2Seq()

    def training_step(self, model, inputs):
        # 调用原始的 training_step
        loss = super().training_step(model, inputs)
        
        # 获取当前学习率
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        # 使用 W&B 记录学习率
        wandb.log({"learning_rate": current_lr}, step=self.state.global_step)
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # breakpoint()
        
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])
        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss
        # breakpoint()
        
        # ********************Accuracy**************************
        # Example accuracy calculation for predictions
        pred_labels = inputs['pred']['labels']  # Assuming true labels are here
        pred_preds = torch.argmax(pred_outputs.logits, dim=-1)
        pred_accuracy = (pred_preds == pred_labels).float().mean()

        # Example accuracy calculation for explanations (if applicable)
        expl_labels = inputs['expl']['labels']  # Assuming true labels for explanations
        expl_preds = torch.argmax(expl_outputs.logits, dim=-1)
        expl_accuracy = (expl_preds == expl_labels).float().mean()
        wandb.log({'train/loss': loss, 
                   'train/loss_pred': pred_outputs.loss, 
                   'train/loss_pred * alpha': self.alpha * pred_outputs.loss,
                   'train/loss_expl': expl_outputs.loss,
                   'train/pred_accuracy': pred_accuracy.item(),  # Logging prediction accuracy
                    'train/expl_accuracy': expl_accuracy.item(),  # Logging explanation accuracy (if applicable)          
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
        torch.cuda.empty_cache() # 也许可以试一下
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False,
                                               ignore_keys=ignore_keys)        
        loss = self.alpha * pred_outputs[0] * self.weight  + (1 - self.alpha) * expl_outputs[0]
        wandb.log({'eval/loss': loss, 
                   'eval/loss_pred': pred_outputs[0], 
                   'eval/loss_pred * alpha': self.alpha * pred_outputs[0],
                   'eval/loss_expl': expl_outputs[0]                  
                   },
                  step=self.state.global_step)
        torch.cuda.empty_cache() # 也许可以试一下
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )