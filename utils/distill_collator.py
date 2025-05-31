import pandas as pd
from transformers import DataCollatorForLanguageModeling


class DistillDataCollator(DataCollatorForLanguageModeling):

    def __call__(self, features, return_tensors=None):

        features_df = pd.DataFrame(features)

        pred_features_df = features_df[['labels', 'input_ids', 'attention_mask']]
        expl_features_df = features_df[['labels_expl', 'input_ids_expl', 'attention_mask_expl']].rename(
            columns={
                "input_ids_expl": "input_ids",
                "attention_mask_expl": "attention_mask",
                "labels_expl": "labels"
            })

        return {
            'pred': super().__call__(pred_features_df.to_dict('records'), return_tensors),
            'expl': super().__call__(expl_features_df.to_dict('records'), return_tensors),
        }
