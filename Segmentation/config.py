
import os
import monai.networks.nets as monai_nets

CONFIG = {
    
    "dataset": {
        "dataset_dir": "/home/tibia/Projet_Hemorragie/Datasets/mbh/Split_Final_Stratified", #"/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI" ,  # ,#/home/tibia/Projet_Hemorragie/MBH_train" #"/home/tibia/Projet_Hemorragie/MBH_train" , "/home/tibia/Projet_Hemorragie/split_in_house_"   #"/home/tibia/Projet_Hemorragie/Split_MBH_V2" ,#/home/tibia/Projet_Hemorragie/MBH_train",             
        "save_dir": "/home/tibia/Projet_Hemorragie/Checkpoints/mbh/test_seg_ntransf",
        "name_run": "vf_split_2021_10_11_15_00_00",
    },
    
    "training": {
        "batch_size": 2,
        "num_epochs": 1000,
        "optimizer": "sgd",  # AdamW or SGD
        "learning_rate": 1e-3,
        "weight_decay": 3e-5,
        "momentum": 0.99,
    },
    
    "scheduler": {
    "type": "linear_warmup",
    "num_warmup_steps": 0  
    },
    
    # "model": {
    #     "spatial_dims": 3,
    #     "in_channels": 1,
    #     "out_channels": 6,
    #     "channels": (32, 64, 128, 256, 320, 320),
    #     "strides": (2, 2, 2, 2, 2),
    #     "num_res_units": 2,
    # },
    
    "augmentation": {
        "spatial_size": (64, 64, 64),
        "pos_ratio": 5.0,
        "neg_ratio": 1.0,
        "prob": 0.5,
        "pixdim": (1.0, 1.0, 1.0),
        "non_zero_norm": True,
        "num_samples": 2,
        "class_ratios_per_dataset": [ 0.1, 0.18, 0.18, 0.18, 0.18, 0.18 ]
    },
    "callbacks": {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1,
        "patience": 50,
    },
    
    "CHECKPOINT_PATH":  "/home/tibia/Projet_Hemorragie/Checkpoints/mbh/test_seg_ntransf/best_model.ckpt",

}
