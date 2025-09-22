
import os
import monai.networks.nets as monai_nets

CONFIG = {
    "dataset": {
        "dataset_dir": "/home/tibia/Projet_Hemorragie/Split_MBH_V2",
        "save_dir": "/home/tibia/Projet_Hemorragie/MBH_seg_v2_test_1",
    },
    "training": {
        "batch_size": 2,
        "num_epochs": 1000,
        "optimizer": "SGD",  # AdamW or SGD
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "momentum": 0.9,
    },
    "model": {
        
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 6,
        "channels": (32, 64, 128, 256, 320, 320),
        "strides": (2, 2, 2, 2, 2),
        "num_res_units": 2,
    },
    "augmentation": {
        "spatial_size": (96, 96, 96),
        "pos_ratio": 5.0,
        "neg_ratio": 1.0,
        "prob": 0.5,
    },
    "callbacks": {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1,
        "patience": 50,
    }
}
