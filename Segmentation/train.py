from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import data.dataset as dataset
import data.transform as T_seg

import config as cfg
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
cfg=cfg.CONFIG
    # Logger W&B
    # ---------------------------
wandb_logger = WandbLogger(project="segmentation_MBH", config=cfg)

    # ---------------------------
    # Load data
    # ---------------------------
train_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/train/img",
                                         f"{cfg['dataset']['dataset_dir']}/train/seg")
val_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/val/img",
                                       f"{cfg['dataset']['dataset_dir']}/val/seg")
    
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

    # ---------------------------
    # Model
    # ---------------------------
num_steps = len(train_loader) * cfg['training']['num_epochs']
model = HemorrhageModel(num_steps=num_steps) # cfg is global

    # ---------------------------
    # Callbacks
    # ---------------------------
callbacks = [
        EarlyStopping(monitor=cfg['callbacks']['monitor'], patience=cfg['callbacks']['patience']),
        ModelCheckpoint(
            monitor=cfg['callbacks']['monitor'],
            mode=cfg['callbacks']['mode'],
            save_top_k=cfg['callbacks']['save_top_k'],
            dirpath=os.path.join(cfg['dataset']['save_dir'], "checkpoints"),
            filename="best_model"
        ),
        #LearningRateMonitor(logging_interval='step')
    ]

    # ---------------------------
    # Trainer
    # ---------------------------
trainer = pl.Trainer(
        max_epochs=cfg['training']['num_epochs'],
        accelerator="auto",
        devices="auto",
        default_root_dir=cfg['dataset']['save_dir'],
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0
    )

    # ---------------------------
    # Train
    # ---------------------------
trainer.fit(model, train_loader, val_loader)







# def train_model(cfg):
#     # ---------------------------
#     # Logger W&B
#     # ---------------------------
#     wandb_logger = WandbLogger(project="segmentation_sweep", config=cfg)

#     # ---------------------------
#     # Load data
#     # ---------------------------
#     train_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/train/img",
#                                          f"{cfg['dataset']['dataset_dir']}/train/seg")
#     val_files = dataset.get_data_files(f"{cfg['dataset']['dataset_dir']}/val/img",
#                                        f"{cfg['dataset']['dataset_dir']}/val/seg")
    
#     train_dataset = PersistentDataset(
#         train_files,
#         transform=T_seg.get_train_transforms(cfg),
#         cache_dir=os.path.join(cfg['dataset']['save_dir'], "cache_train")
#     )

#     val_dataset = PersistentDataset(
#         val_files,
#         transform=T_seg.get_val_transforms(cfg),
#         cache_dir=os.path.join(cfg['dataset']['save_dir'], "cache_val")
#     )

#     train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8)
#     val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=8)

#     # ---------------------------
#     # Model
#     # ---------------------------
#     num_steps = len(train_loader) * cfg['training']['num_epochs']
#     model = HemorrhageModel(num_steps=num_steps, cfg=cfg)  # tu passes cfg pour optimizer/scheduler dynamiques

#     # ---------------------------
#     # Callbacks
#     # ---------------------------
#     callbacks = [
#         EarlyStopping(monitor=cfg['callbacks']['monitor'], patience=cfg['callbacks']['patience']),
#         ModelCheckpoint(
#             monitor=cfg['callbacks']['monitor'],
#             mode=cfg['callbacks']['mode'],
#             save_top_k=cfg['callbacks']['save_top_k'],
#             dirpath=cfg['dataset']['save_dir'],
#             filename="best_model"
#         ),
#         LearningRateMonitor(logging_interval='step')
#     ]

#     # ---------------------------
#     # Trainer
#     # ---------------------------
#     trainer = pl.Trainer(
#         max_epochs=cfg['training']['num_epochs'],
#         accelerator="auto",
#         devices="auto",
#         default_root_dir=cfg['dataset']['save_dir'],
#         logger=wandb_logger,
#         callbacks=callbacks,
#         gradient_clip_val=1.0
#     )

#     # ---------------------------
#     # Train
#     # ---------------------------
#     trainer.fit(model, train_loader, val_loader)