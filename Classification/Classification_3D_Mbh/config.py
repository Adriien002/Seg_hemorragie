import os
import torch
# ======================
# CONFIGURATION
# ======================

CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
NUM_CLASSES = 1 #6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
num_epochs = 1000
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_classif3D"
os.makedirs(SAVE_DIR, exist_ok=True)