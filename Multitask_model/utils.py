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

import torch
import itertools

def extract_sliding_window_patches(image_tensor, label, roi_size=(96, 96, 96), overlap=0.25):
    """
    Extrait patches avec sliding window.
    
    Args:
        image_tensor: Tensor [C, H, W, D]
        label: Label de classification
        roi_size: Taille des patches
        overlap: Pourcentage de chevauchement (0.0 à 0.99)
    
    Returns:
        Liste de dicts avec 'image', 'label', 'task'
    """
    C, H, W, D = image_tensor.shape
    
    # Calcule le stride
    stride = tuple(int(r * (1 - overlap)) for r in roi_size)
   
    
    # # Génère les positions de départ pour chaque dimension
    def get_starts(dim_size, roi_dim, stride_dim):
        starts = list(range(0, dim_size - roi_dim + 1, stride_dim))
        # Assure la couverture complète
        if starts[-1] + roi_dim < dim_size:
            starts.append(dim_size - roi_dim)
        return starts
    
    h_starts = get_starts(H, roi_size[0], stride[0])
    w_starts = get_starts(W, roi_size[1], stride[1])
    d_starts = get_starts(D, roi_size[2], stride[2])

    
    patches = []
    for h, w, d in itertools.product(h_starts, w_starts, d_starts):
        patch = image_tensor[
            :,
            h:h + roi_size[0],
            w:w + roi_size[1],
            d:d + roi_size[2]
        ]
        
        patches.append({
            "image": patch,
            "label": label,
            "task": "classification"
        })
        
         

    

    return patches
