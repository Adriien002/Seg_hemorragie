
import os
import monai.networks.nets as monai_nets

DATASET_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_aug_low"
os.makedirs(SAVE_DIR, exist_ok=True)
batch_size = 2
num_epochs = 1000
model = monai_nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
        )
spatial_size = (96, 96, 96) # Taille des patches pour l'entraînement
pos_ratio = 5.0  # Ratio pour le recadrage positif
neg_ratio = 1.0  # Ratio pour le recadrage négatif