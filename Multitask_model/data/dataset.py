
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
SEG_IN_HOUSE_DIR = config.IN_HOUSE_DIR
# ======================
# DATA PREPARATION
# ======================

def _seg_fname_to_pid(fname: str) -> str:
    """Extrait le patientID_studyID depuis un nom de fichier seg.

    Deux conventions dans Split_Final_Stratified :
      MBH_SEG_2025_LLG_2025_06_12_ID_xxx_ID_yyy.nii.gz  -> ID_xxx_ID_yyy
      MBH_v2_val_5_ID_xxx_ID_yyy.nii.gz                 -> MBH_v2_val_5_ID_xxx_ID_yyy
    """
    base = fname.replace(".nii.gz", "")
    if base.startswith("MBH_SEG_2025_LLG_"):
        parts = base.split("_")
        for i, p in enumerate(parts):
            if p == "ID" and i > 0:
                return "_".join(parts[i:])
    return base


def get_segmentation_data(split="train"):
    img_dir = Path(SEG_DIR) / split / "img"
    seg_dir = Path(SEG_DIR) / split / "seg"

    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    assert len(images) == len(labels), "Mismatch between image and label counts"

    # Vrais labels de classification depuis le CSV seg-derive
    csv_path = (Path(CLASSIFICATION_DATA_DIR) / "splits"
                / f"classif_annotation_seg{split}.csv")
    id_to_cls: dict = {}
    if csv_path.exists():
        df_cls = pd.read_csv(csv_path)
        for _, row in df_cls.iterrows():
            id_to_cls[str(row["patientID_studyID"])] = torch.tensor(
                [float(row[c]) for c in SEG_LABEL_COLS], dtype=torch.float32
            )

    data = []
    for img, lbl in zip(images, labels):
        pid = _seg_fname_to_pid(img.name)
        if pid in id_to_cls:
            data.append({
                "image":       str(img),
                "label":       str(lbl),
                "class_label": id_to_cls[pid],
                "task":        "seg_multi",
            })
        else:
            data.append({
                "image":       str(img),
                "label":       str(lbl),
                "class_label": torch.zeros(6, dtype=torch.float32),
                "task":        "seg_orig",
            })

    n_multi = sum(1 for d in data if d["task"] == "seg_multi")
    print(f"{split.upper()} seg — seg_multi: {n_multi} | seg_orig: {len(data)-n_multi} | total: {len(data)}")
    return data

def get_inhouse_segmentation_data(split="train"):
    # Utilise le préfixe défini dans config (ex: "/.../split_in_house_" + "train" = "/.../split_in_house_train")
  
    
    img_dir = Path(SEG_IN_HOUSE_DIR) / split / "img"
    seg_dir = Path(SEG_IN_HOUSE_DIR) / split / "seg"
    
    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    
    assert len(images) == len(labels), f"Mismatch between In-House image and label counts for split {split}"

    data = []
    for img, lbl in zip(images, labels):
        data.append({
            "image": str(img),
            "label": str(lbl),
            "class_label": torch.tensor([0.0] * 6, dtype=torch.float32), 
            "task": "seg_inhouse"  # <-- NOUVELLE TÂCHE
        })
        
    return data


def get_instance_segmentation_data(split="train"):
    img_dir = Path(config.INSTANCE_DIR) / split / "img"
    seg_dir = Path(config.INSTANCE_DIR) / split / "seg"

    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))

    assert len(images) == len(labels), f"Mismatch between INSTANCE image and label counts for split {split}"

    data = []
    for img, lbl in zip(images, labels):
        data.append({
            "image": str(img),
            "label": str(lbl),
            "class_label": torch.tensor([0.0] * 6, dtype=torch.float32),
            "task": "seg_instance"
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



