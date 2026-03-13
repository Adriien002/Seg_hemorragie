
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import config
import torch

SEG_LABEL_COLS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

SEG_DIR = config.SEG_DIR
CLASSIFICATION_DATA_DIR = config.CLASSIFICATION_DATA_DIR
SAVE_DIR = config.SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
# ======================
# DATA PREPARATION
# ======================
def get_segmentation_data(split="train"):
    img_dir = Path(SEG_DIR) / split / "img"
    seg_dir = Path(SEG_DIR) / split / "seg"
    
    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    
    assert len(images) == len(labels), "Mismatch between image and label counts"

    data = []
    for img, lbl in zip(images, labels):
        data.append({
            "image": str(img),
            "label": str(lbl),
            "class_label": torch.tensor([0.0] * 6, dtype=torch.float32),  # Pas de label global pour la tête de segmentation
            "task": "segmentation"
        })
        
    return data



def get_classification_data(split="train"):
    nii_dir   = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    # ── Positifs ─────────────────────────────────────────────────────────────
    df = pd.read_csv(nii_dir / "splits" / f"{split}_split.csv")
    positives = []
    for _, row in df.iterrows():
        pid = row['patientID_studyID']
        mask_path = str(Path(config.PSEUDO_MASKS_DIR) / f"{pid}.nii.gz")
        if not os.path.exists(mask_path):
            continue  # nnU-Net a raté ce cas, on skip
        positives.append({
            "image":       str(nii_dir / f"{pid}.nii.gz"),
            "label":       mask_path,
            "class_label": torch.tensor(
                       np.array([row[col] for col in label_cols], dtype=np.float32)
                   ),
            "has_mask":    True,
            "task":        "classification"
        })

    # ── Sains ────────────────────────────────────────────────────────────────
    healthy_df = pd.read_csv(nii_dir / "splits" / f"{split}_healthy_split.csv")
    negatives = [{
        "image":       str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label":       None,
        "class_label":  torch.tensor([0.0] * 6, dtype=torch.float32),
        "has_mask":    False,
        "task":        "classification"
    } for _, row in healthy_df.iterrows()]

    data = positives + negatives
    np.random.default_rng(seed=42).shuffle(data)

    print(f"{split.upper()} cls — positifs: {len(positives)} | sains: {len(negatives)} | total: {len(data)}")
    return data


def compute_pos_weights(split="train"):
    """pos_weight[i] = n_neg_i / n_pos_i — pour BCEWithLogitsLoss"""
    data   = get_classification_data(split)
    labels = np.array([d["class_label"] for d in data])   # (N, 6)
    n_pos  = labels.sum(axis=0)
    n_neg  = len(labels) - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1)

    print("pos_weights :")
    for name, w, p, n in zip(config.CLASS_NAMES, pos_weight, n_pos, n_neg):
        print(f"  {name:25s}: {w:.2f}  ({int(p)} pos / {int(n)} neg)")

    return torch.tensor(pos_weight, dtype=torch.float32)




def get_optimized_classification_data(split="train", seg_count=154):
    csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # Identifier les cas positifs (au moins une hémorragie)
    df['is_positive'] = df[label_cols].sum(axis=1) > 0
    positives = df[df['is_positive']]
    negatives = df[~df['is_positive']]
    
    # Stratégie différente pour train/val
    
        # TRAIN : On garde TOUS les positifs (371) même si >2x seg_count
    selected_pos = positives
    needed_neg = max(0, 2*seg_count - len(selected_pos))
    selected_neg = negatives.sample(min(needed_neg, len(negatives)), random_state=42) if needed_neg > 0 else pd.DataFrame()
    
    
    balanced_df = pd.concat([selected_pos, selected_neg]).sample(frac=1, random_state=42)
    
    # Statistiques
    print(f"{split.upper()} - Positifs: {len(selected_pos)}/{len(positives)} | Négatifs: {len(selected_neg)}")
    
    return [{
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32),
        "task": "classification"
    } for _, row in balanced_df.iterrows()]
    
    
def get_multitask_dataset(split="train"):
    seg_data = get_segmentation_data(split)
    cls_data = get_classification_data(split)
    return seg_data + cls_data



def get_balanced_multitask_dataset(split="train"):
    seg_data = get_segmentation_data(split)
    seg_count = len(seg_data)
    cls_data = get_optimized_classification_data(split, seg_count)
    print(f"Nombre de données de segmentation : {len(seg_data)}")
    print(f"Nombre de données de classification : {len(cls_data)}")
    return seg_data + cls_data


def get_equalized_multitask_dataset(split="train"):
    seg_data = get_segmentation_data(split)
    seg_count = len(seg_data)

    csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    # Identifier positifs/négatifs
    df['is_positive'] = df[label_cols].sum(axis=1) > 0
    positives = df[df['is_positive']]
    negatives = df[~df['is_positive']]

    # Équilibrer : même nombre de négatifs que de positifs
    n_pos = len(positives)
    selected_pos = positives
    selected_neg = negatives.sample(n=min(n_pos, len(negatives)), random_state=42)

    balanced_df = pd.concat([selected_pos, selected_neg]).sample(frac=1, random_state=42)

    print(f"{split.upper()} - Positifs: {len(selected_pos)} | Négatifs: {len(selected_neg)}")

    cls_data = [{
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32),
        "task": "classification"
    } for _, row in balanced_df.iterrows()]

    print(f"Nombre de données de segmentation : {seg_count}")
    print(f"Nombre de données de classification équilibrées : {len(cls_data)}")

    return seg_data + cls_data
