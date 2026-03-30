
import os
import torch
# ======================
# CONFIGURATION
# ======================
SEG_DIR = '/home/tibia/Projet_Hemorragie/Datasets/mbh/Split_Final_Stratified'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/Datasets/mbh/MBH_label_case'
PSEUDO_MASKS_DIR = '/home/tibia/Projet_Hemorragie/Datasets/mbh/Pseudo_Masks' 
NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
num_epochs = 1000
batch_size = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/tibia/Projet_Hemorragie/Checkpoints/mbh/full_in_house_96666"
os.makedirs(SAVE_DIR, exist_ok=True)
IN_HOUSE_DIR = "/home/tibia/Projet_Hemorragie/Datasets/in_house/split_in_house_"