
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

SEG_LABEL_COLS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_pos_cases"
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
            "task": "segmentation"
        })
        
    return data


def get_classification_data(split="train"):
    csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    data = []
    for _, row in df.iterrows():
        image_path = str(nii_dir / f"{row['patientID_studyID']}.nii.gz")
        label = np.array([row[col] for col in label_cols], dtype=np.float32)
        
        data.append({
            "image": image_path,
            "label": label,
            "task": "classification"
        })
    return data


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