from monai.data.utils import list_data_collate
from collections.abc import Iterable
import torch
import pandas as pd

def flatten(batch):
    for item in batch:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def multitask_collate_fn(batch):
    flat_batch = list(flatten(batch))  

    classification_batch = []
    segmentation_batch = []

    for item in flat_batch:
        if item["task"] == "classification":
            classification_batch.append(item)
        elif item["task"] == "segmentation":
            segmentation_batch.append(item)
           
        else:
            raise ValueError(f"Tâche inconnue : {item['task']}")

    result = {
        "classification": list_data_collate(classification_batch) if classification_batch else None,
        "segmentation": list_data_collate(segmentation_batch) if segmentation_batch else None
    }
    
    return result


def calculate_pos_weights(csv_path, label_cols):
    """Calculate pos_weight for BCEWithLogitsLoss based on class frequency."""
    df = pd.read_csv(csv_path)
    pos_weights = []
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count
        if pos_count > 0:
            pos_weight = neg_count / pos_count
        else:
            pos_weight = 1.0  # Default weight if no positive samples
        pos_weights.append(pos_weight)
    return torch.tensor(pos_weights, dtype=torch.float)
