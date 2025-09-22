import torch

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    assert (isinstance(logits, torch.Tensor))
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels
