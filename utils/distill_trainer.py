import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Trainer


class DistillTrainer(Trainer):

    def __init__(self, alpha, log_compute_loss_func=None, log_pred_step_func=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.log_compute_loss_func = log_compute_loss_func
        self.log_pred_step_func = log_pred_step_func

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # This is for the case when we launch ordinary compute loss for the prediction step.
        if 'pred' not in inputs and 'expl' not in inputs:
            return super().compute_loss(model, inputs,
                                        return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        current_lr = self.optimizer.param_groups[0]["lr"]

        pred_labels = inputs['pred']['labels']  # Assuming true labels are here
        pred_preds = torch.argmax(pred_outputs.logits, dim=-1)
        pred_accuracy = (pred_preds == pred_labels).float().mean()

        expl_labels = inputs['expl']['labels']  # Assuming true labels for explanations
        expl_preds = torch.argmax(expl_outputs.logits, dim=-1)
        expl_accuracy = (expl_preds == expl_labels).float().mean()

        if self.log_compute_loss_func is not None:
            self.log_compute_loss_func(
                data={
                    'train/loss': loss,
                    'train/loss_pred': pred_outputs.loss,
                    'train/loss_expl': expl_outputs.loss,
                    'train/pred_accuracy': pred_accuracy.item(),  # Logging prediction accuracy
                    'train/expl_accuracy': expl_accuracy.item(),  # Logging explanation accuracy (if applicable)
                    'learning_rate': current_lr,
                },
            )

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model,
                                               inputs=inputs['pred'],
                                               prediction_loss_only=False,
                                               ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model,
                                               inputs=inputs['expl'],
                                               prediction_loss_only=False,
                                               ignore_keys=ignore_keys)

        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_outputs[0]

        if self.log_pred_step_func is not None:
            self.log_pred_step_func(
                data={
                    'eval/loss': loss,
                    'eval/loss_pred': pred_outputs[0],
                    'eval/loss_expl': expl_outputs[0]
                },
            )

        return (loss, pred_outputs[1], pred_outputs[2])
