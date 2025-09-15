from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import data.dataset as dataset
import data.transform as T_seg

import config
import wandb
from models.lightning import HemorrhageModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

#initialisaiton w&b
wandb.init(project="segmentation_sweep", config=config)
cfg = wandb.config

#Logger
wandb_logger = WandbLogger(project="segmentation_sweep",config=config)






# Load data (same as original)
train_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/train/img",
                                     f"{cfg['dataset']['dataset_dir']}/train/seg")
val_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/val/img",
                                   f"{cfg['dataset']['dataset_dir']}/val/seg")
test_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/test/img",
                                    f"{cfg['dataset']['dataset_dir']}/test/seg"))
    
    
test_dataset = PersistentDataset(
        test_files,
        transform=T_seg.val_transforms,
        cache_dir=os.path.join(cfg['dataset']['save_dir'], "cache_test")  
    )
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=4  )

train_dataset = PersistentDataset(
        train_files,
        transform=T_seg.get_train_transforms(cfg),
        cache_dir=os.path.join(cfg['dataset']['save_dir'], "cache_train")
    )

val_dataset = PersistentDataset(
        val_files,
        transform=T_seg.get_val_transforms(cfg),
        cache_dir=os.path.join(cfg['dataset']['save_dir'], "cache_val")
    )

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

# Initialize model with checkpoint if available
model = HemorrhageModel(num_steps=len(train_loader) * cfg['training']["num_epochs"])
print(f"Total number of steps : {len(train_loader) * cfg['training']["num_epochs"]}")

# Callbacks
callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["callbacks"]["patience"])
    ]



# Configure trainer with progress bar and checkpointing
trainer = pl.Trainer(
        max_epochs=cfg['training']['num_epochs'],
        #check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[0],
        default_root_dir=cfg['dataset']['save_dir'],
        logger= wandb_logger, # Dossier où sont stockés les logs
        #accumulate_grad_batches=4  # Accumulate gradients over 4 batches
        callbacks=callbacks,
        
    )

    # Start training
trainer.fit(model, train_loader, val_loader)
    
    # #Test
    # print("Starting test phase...")
    # trainer.test(model, dataloaders=test_loader)

