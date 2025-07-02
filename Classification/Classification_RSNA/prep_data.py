# =======================
#  Imports
# =======================
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pydicom
import monai.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

# =======================
#  Function definitions
# =======================

def get_dicom_path(csv_path, dicom_dir, i):
    """
    Return the i-th DICOM file path listed in the CSV.
    """
    df = pd.read_csv(csv_path)
    filename = df.iloc[i]['filename']
    return Path(dicom_dir) / filename

def get_id(dcm_path):
    """
    Extract the SOPInstanceUID (unique ID) from the DICOM metadata.
    """
    dcm = pydicom.dcmread(dcm_path)
    return getattr(dcm, "SOPInstanceUID", None)

def window_image(img, center, width):
    """
    Apply windowing to the raw pixel array using the specified window center and width.
    """
    img_min = center - width // 2
    img_max = center + width // 2
    return np.clip(img, img_min, img_max)

def load_dicom_tensor(csv_path, dicom_dir, i):
    """
    Load the i-th DICOM image as a tensor using MONAI's LoadImage.
    Also return its metadata.
    """
    dcm_path = get_dicom_path(csv_path, dicom_dir, i)
    loader = T.LoadImage(image_only=False)  # returns (image, metadata)
    image, meta = loader(str(dcm_path))
    return image, meta, dcm_path
# =======================
# Main processing loop
# =======================
def main():
    """
    Main function to process DICOM images from the specified CSV and directory.
  
    """
    # Define paths
    csv_path = "/home/tibia/Projet_Hemorragie/Classification/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/data/csv/train_fold0.csv"
    dicom_dir = "/home/tibia/Projet_Hemorragie/Classification/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/data/rsna-intracranial-hemorrhage-detection/stage_2_train"

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Resize transform with MONAI
    resize_transform = T.Resize((224, 224))


    for i in tqdm(range(5)):
        # Load image and metadata
        image, meta, dcm_path = load_dicom_tensor(csv_path, dicom_dir, i)

        # Load DICOM header manually for additional metadata
        dicom_raw = pydicom.dcmread(str(dcm_path))
        center = int(dicom_raw.WindowCenter)
        width = int(dicom_raw.WindowWidth)
        slope = float(dicom_raw.RescaleSlope)
        intercept = float(dicom_raw.RescaleIntercept)

        # Apply rescale slope and intercept
        img_np = image.numpy()
        img_np = img_np * slope + intercept

        # Apply windowing
        img_windowed = window_image(img_np, center, width)

        # Normalize [0, 1] and convert to tensor
        img_norm = (img_windowed - img_windowed.min()) / (img_windowed.max() - img_windowed.min())
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: [1, H, W]

        # Resize to 224x224
        resized_tensor = resize_transform(img_tensor)

        #stocker les images dans un dossier ? un cache ? 

    if __name__ == "__main__":
        main()