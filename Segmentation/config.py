
import os
import monai.networks.nets as monai_nets

#Configurations
DATASET_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_aug_low"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "best_model.ckpt")

# Training parameters
batch_size = 2
num_epochs = 1000

# Model parameters
model = monai_nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
        )

# Data augmentation parameters
spatial_size = (96, 96, 96) # Taille des patches pour l'entraînement
pos_ratio = 5.0  # Ratio pour le recadrage positif
neg_ratio = 1.0  # Ratio pour le recadrage négatif


#Callbacks parameters
monitor = "val_loss"
mode = "min"
save_top_k = 2
patience = 50  # Nombre d'époques sans amélioration avant d'arrêter