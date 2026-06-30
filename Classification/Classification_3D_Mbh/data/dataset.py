from pathlib import Path
import pandas as pd
import numpy as np
import torch
import config

LABEL_COLS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


def _seg_fname_to_pid(fname: str) -> str:
    base = fname.replace(".nii.gz", "")
    if base.startswith("MBH_SEG_2025_LLG_"):
        parts = base.split("_")
        for i, p in enumerate(parts):
            if p == "ID" and i > 0:
                return "_".join(parts[i:])
    return base


def get_classification_data(split: str = "train"):
    """Données MBH avec labels de classification.
    Positifs : image + pseudo-masque (crop guidé sur la lésion).
    Sains    : image seule (crop aléatoire).
    """
    nii_dir = Path(config.CLASSIFICATION_DATA_DIR)

    df = pd.read_csv(nii_dir / "splits" / f"{split}_split.csv")
    positives = []
    for _, row in df.iterrows():
        pid = row['patientID_studyID']
        mask_path = Path(config.PSEUDO_MASKS_DIR) / f"{pid}.nii.gz"
        if not mask_path.exists():
            continue
        positives.append({
            "image":       str(nii_dir / f"{pid}.nii.gz"),
            "label":       str(mask_path),
            "class_label": torch.tensor(
                np.array([row[col] for col in LABEL_COLS], dtype=np.float32)
            ),
            "has_mask": True,
        })

    healthy_df = pd.read_csv(nii_dir / "splits" / f"{split}_healthy_split.csv")
    negatives = [{
        "image":       str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "class_label": torch.zeros(6, dtype=torch.float32),
        "has_mask":    False,
    } for _, row in healthy_df.iterrows()]

    data = positives + negatives
    np.random.default_rng(seed=42).shuffle(data)
    print(f"{split.upper()} cls       — positifs: {len(positives)} | sains: {len(negatives)} | total: {len(data)}")
    return data


def get_seg_classification_data(split: str = "train"):
    """Labels de classification extraits du jeu de segmentation MBH.
    Seuls les cas ayant une annotation dans classif_annotation_seg{split}.csv sont retenus.
    Le vrai masque de segmentation est utilisé pour guider le crop (même stratégie
    que le modèle multi-tâche), mais n'est jamais supervisé ici.
    """
    img_dir = Path(config.SEG_DIR) / split / "img"
    seg_dir = Path(config.SEG_DIR) / split / "seg"

    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    assert len(images) == len(labels), "Mismatch image/label dans SEG_DIR"

    csv_path = (Path(config.CLASSIFICATION_DATA_DIR) / "splits"
                / f"classif_annotation_seg{split}.csv")
    if not csv_path.exists():
        print(f"AVERTISSEMENT : {csv_path} introuvable, get_seg_classification_data retourne []")
        return []

    df_cls = pd.read_csv(csv_path)
    id_to_cls = {
        str(row["patientID_studyID"]): torch.tensor(
            [float(row[c]) for c in LABEL_COLS], dtype=torch.float32
        )
        for _, row in df_cls.iterrows()
    }

    data = []
    for img, lbl in zip(images, labels):
        pid = _seg_fname_to_pid(img.name)
        if pid not in id_to_cls:
            continue  # pas d'annotation de classification → on skip
        data.append({
            "image":       str(img),
            "label":       str(lbl),   # vrai masque seg → crop guidé uniquement
            "class_label": id_to_cls[pid],
            "has_mask":    True,
        })

    print(f"{split.upper()} seg→cls   — cas annotés : {len(data)}")
    return data


def compute_pos_weights(split: str = "train") -> torch.Tensor:
    """pos_weight[i] = n_neg_i / n_pos_i pour BCEWithLogitsLoss.
    Calculé sur l'union des deux sources de données.
    """
    data = get_classification_data(split) + get_seg_classification_data(split)
    labels = np.array([d["class_label"].numpy() for d in data])
    n_pos  = labels.sum(axis=0)
    n_neg  = len(labels) - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1)

    print("pos_weights :")
    for name, w, p, n in zip(config.CLASS_NAMES, pos_weight, n_pos, n_neg):
        print(f"  {name:25s}: {w:.2f}  ({int(p)} pos / {int(n)} neg)")

    return torch.tensor(pos_weight, dtype=torch.float32)
