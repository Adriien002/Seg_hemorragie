from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import config


def get_classification_data(split="train"):
    csv_path = Path(config.CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(config.CLASSIFICATION_DATA_DIR)
 
    
    data = []
    for _, row in df.iterrows():
        image_path = str(nii_dir / f"{row['patientID_studyID']}.nii.gz")
        label = np.array([row[col] for col in config.CLASS_NAMES], dtype=np.float32)
        
        data.append({
            "image": image_path,
            "label": label})
    return data
