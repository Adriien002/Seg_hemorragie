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


def main():
    cfg=config.CONFIG
        # Logger W&B
        # ---------------------------
    wandb_logger = WandbLogger(project="segmentation_MBH", config=cfg,save_dir=cfg['dataset']['save_dir'])

        # ---------------------------
        # Load data
        # ---------------------------
    #train_files = dataset.get_data_files_3(f"{cfg['dataset']['dataset_dir']}")
    
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
    # callbacks = [
    #         #EarlyStopping(monitor=cfg['callbacks']['monitor'], patience=cfg['callbacks']['patience']),
    #         ModelCheckpoint(
    #             monitor=cfg['callbacks']['monitor'],
    #             mode=cfg['callbacks']['mode'],
    #             save_top_k=cfg['callbacks']['save_top_k'],
    #             dirpath=os.path.join(cfg['dataset']['save_dir'], "checkpoints"),
    #             filename="best_model"
    #         )
            #LearningRateMonitor(logging_interval='step')
       # ]
    callbacks = [
    ModelCheckpoint(
        dirpath=os.path.join(cfg['dataset']['save_dir'], "checkpoints"),
        filename="last_model",
        save_last=True,      # garde uniquement le dernier modèle
        save_top_k=1      
    )
]


        # ---------------------------
        # Trainer
        # ---------------------------
    trainer = pl.Trainer(
            max_epochs=cfg['training']['num_epochs'],
            accelerator="auto",
            devices=[1],
            default_root_dir=cfg['dataset']['save_dir'],
            logger=wandb_logger,
            callbacks=callbacks
            #gradient_clip_val=1.0
        )

        # ---------------------------
        # Train
   
    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        # NETTOYAGE EXPLICITE 
        wandb.finish()  
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        
        
        if hasattr(train_dataset, 'shutdown'):
            train_dataset.shutdown()
        if hasattr(val_dataset, 'shutdown'):
            val_dataset.shutdown()

if __name__ == "__main__":
    main()

