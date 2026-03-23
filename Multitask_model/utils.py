from monai.data.utils import list_data_collate
from collections.abc import Iterable
import torch
import pandas as pd
import numpy as np


def flatten(batch):
    for item in batch:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


# def _sanitize_item(item: dict) -> dict:
#     """
#     Convertit les champs non-MetaTensor en torch.Tensor purs
#     pour éviter que list_data_collate ne panique sur les métadonnées.
#     """
#     out = {}
#     for k, v in item.items():
#         if isinstance(v, np.ndarray):
#             out[k] = torch.from_numpy(v)
#         elif isinstance(v, (int, float)):
#             out[k] = torch.tensor(v)
#         else:
#             out[k] = v  # MetaTensor, str, etc. → inchangé
#     return out


# def multitask_collate_fn(batch):
#     flat_batch = list(flatten(batch))

#     classification_batch = []
#     segmentation_batch = []

#     for item in flat_batch:
#         task = item["task"]
#         sanitized = _sanitize_item(item)
        
#         if task == "classification":
#             classification_batch.append(sanitized)
#         elif task == "segmentation":
#             segmentation_batch.append(sanitized)
#         else:
#             raise ValueError(f"Tâche inconnue : {task}")

#     result = {
#         "classification": list_data_collate(classification_batch) if classification_batch else None,
#         "segmentation": list_data_collate(segmentation_batch) if segmentation_batch else None,
#     }
#     return result
from monai.data.utils import list_data_collate
import torch

def multitask_collate_fn(batch):
    flat_batch = list(flatten(batch))

    classification_batch = []
    segmentation_mbh = []
    segmentation_inhouse = []

    for item in flat_batch:
        task = item["task"]
        
        # 1. On crée un nouveau dictionnaire propre (sans les métadonnées MONAI)
        clean_item = {"task": task}
        
        # 2. Nettoyage de l'image (extraction du tenseur pur)
        img = item["image"]
        clean_item["image"] = img.as_tensor() if hasattr(img, "as_tensor") else img
        
        # 3. Nettoyage du masque de segmentation (uniquement s'il existe et n'est pas None)
        if "label" in item and item["label"] is not None:
            lbl = item["label"]
            clean_item["label"] = lbl.as_tensor() if hasattr(lbl, "as_tensor") else lbl

        # 4. Nettoyage du label de classification
        if "class_label" in item:
            cl = item["class_label"]
            if hasattr(cl, "as_tensor"):
                clean_item["class_label"] = cl.as_tensor()
            elif not isinstance(cl, torch.Tensor):
                clean_item["class_label"] = torch.as_tensor(cl, dtype=torch.float32)
            else:
                clean_item["class_label"] = cl

        # 5. Répartition dans le bon sous-batch
        if task == "classification":
            classification_batch.append(clean_item)
        # elif task == "segmentation":
        #     segmentation_batch.append(clean_item)
        elif task == "seg_orig":
            segmentation_mbh.append(clean_item)
       
        elif task == "seg_inhouse":
            segmentation_inhouse.append(clean_item)
        else:
            raise ValueError(f"Tâche inconnue : {task}")

    # 6. Collation finale en toute sécurité
    result = {
        "classification": list_data_collate(classification_batch) if classification_batch else None,
        "seg_orig": list_data_collate(segmentation_mbh) if segmentation_mbh else None,
        "seg_inhouse":   list_data_collate(segmentation_inhouse)   if segmentation_inhouse   else None,
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


import torch
import itertools
import random

def extract_sliding_window_patches(
    image_tensor,
    label,
    roi_size=(96, 96, 96),
    overlap=0.25,
    max_patches=18  
):
    """
    Extrait un nombre limité de patches avec une fenêtre glissante.
    """
    C, H, W, D = image_tensor.shape

    stride = tuple(int(r * (1 - overlap)) for r in roi_size)

    def get_starts(dim_size, roi_dim, stride_dim):
        starts = list(range(0, dim_size - roi_dim + 1, stride_dim))
        if starts[-1] + roi_dim < dim_size:
            starts.append(dim_size - roi_dim)
        return starts

    h_starts = get_starts(H, roi_size[0], stride[0])
    w_starts = get_starts(W, roi_size[1], stride[1])
    d_starts = get_starts(D, roi_size[2], stride[2])

    # 🔹 Génère toutes les positions possibles
    all_coords = list(itertools.product(h_starts, w_starts, d_starts))

    # 🔹 Si trop de patches, on en prend un sous-ensemble aléatoire
    if len(all_coords) > max_patches:
        all_coords = random.sample(all_coords, max_patches)

    patches = []
    for h, w, d in all_coords:
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