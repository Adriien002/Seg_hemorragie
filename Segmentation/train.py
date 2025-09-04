from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import data.dataset as dataset
import data.transform as T_seg

import config

from models.lightning import HemorrhageModel
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

# adapter à la segmentation -> ici mtsk
config_l = dict(
    sharing_type="soft",   # "soft" ou "fine_tune"
    model="BasicUNetWithClassification",
    loss_weighting="none",
    dataset_size="small : positifs cases",
    batch_size=2,
    learning_rate=1e-3,
    optimizer="sgd",
    seed=42
)

# Génération automatique de tags à partir de config
tags = [f"{k}:{v}" for k, v in config_l.items() if k in ["sharing_type", "optimizer", "model", "loss_weighting"]]

wandb_logger = WandbLogger(
    project="hemorrhage_multitask_test",
    group="noponderation",
    tags=tags,
    config=config_l
)





num_epochs = 1000
    # Load data (same as original)
train_files = dataset.get_data_files(f"{config.DATASET_DIR}/train/img", f"{config.DATASET_DIR}/train/seg")
val_files = dataset.get_data_files(f"{config.DATASET_DIR}/val/img", f"{config.DATASET_DIR}/val/seg")
test_files = dataset.get_data_files(f"{config.DATASET_DIR}/test/img", f"{config.DATASET_DIR}/test/seg")
    
    
test_dataset = PersistentDataset(
        test_files,
        transform=T_seg.val_transforms,
        cache_dir=os.path.join(config.SAVE_DIR, "cache_test")  
    )
    
test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,  
        num_workers=4  )

train_dataset = PersistentDataset(
        train_files,
        transform=T_seg.transforms,
        cache_dir=os.path.join(config.SAVE_DIR, "cache_train")
    )

val_dataset = PersistentDataset(
        val_files,
        transform=T_seg.val_transforms,
        cache_dir=os.path.join(config.SAVE_DIR, "cache_val")
    )

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize model with checkpoint if available
model = HemorrhageModel(num_steps=len(train_loader) * num_epochs)
print(f"Total number of steps : {len(train_loader) * num_epochs}")

    # Configure trainer with progress bar and checkpointing
trainer = pl.Trainer(
        max_epochs=num_epochs,
        #check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[0],
        default_root_dir=config.SAVE_DIR,
        logger= wandb_logger, # Dossier où sont stockés les logs
        #accumulate_grad_batches=4  # Accumulate gradients over 4 batches
        
    )

    # Start training
trainer.fit(model, train_loader, val_loader)
    
    # #Test
    # print("Starting test phase...")
    # trainer.test(model, dataloaders=test_loader)

