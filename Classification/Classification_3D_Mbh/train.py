# Import depuis data/
import data.dataset as dataset
import data.transform as T_mtsk

# Import depuis models/
from models.architecture import resnet
from models.lightning_module import ClassifierModule



from monai.data import DataLoader, PersistentDataset, CacheDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import config

import random

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")
config_l = dict(
    sharing_type="hard",   # "soft" ou "fine_tune"
    model="BasicUNetWithClassification",
    loss_weighting="none",
    dataset_size="balanced",  # "full" ou "balanced" ou "optimized"
    batch_size=2,
    learning_rate=1e-3,
    optimizer="sgd",
    seed=42
)
torch.cuda.set_device(0)
# Génération automatique de tags à partir de config
tags = [f"{k}:{v}" for k, v in config_l.items() if k in ["sharing_type", "optimizer", "model", "loss_weighting"]]

wandb_logger = WandbLogger(
    project="hemorrhage_multitask_test",
    group="noponderation",
    tags=tags,
    config=config_l,
    name="multitask_unet3d"
)

torch.cuda.empty_cache()

train_data= dataset.get_classification_data("train")
val_data= dataset.get_classification_data("val")

train_transforms = T_mtsk.get_train_transforms()
val_transforms = T_mtsk.get_val_transforms

train_dataset = PersistentDataset(
    data=train_data,
    transform=train_transforms, 
    cache_dir="./persistent_cache/3D_train_cache"
)

val_dataset = PersistentDataset(
    data=val_data,
    transform=val_transforms, 
    cache_dir="./persistent_cache/3D_val_cache"
)


train_loader = DataLoader(
        train_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
)

val_loader = DataLoader(
        val_data, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
)




model =ClassifierModule(num_steps=len(train_loader) * config.num_epochs)
    
trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=[0],
        default_root_dir=config.SAVE_DIR,
        logger= wandb_logger,
        #logger=TensorBoardLogger(SAVE_DIR, name="multitask_unet3
        #gradient_clip_val=1.0,  # Gradient clipping pour la stabilité
        log_every_n_steps=50,
        #accumulate_grad_batches=4 ,# Ajout car pour gérer les petites tailles de batch ( dues à limitation mémoire)
        precision='16-mixed',  # Mixed precision pour accélérer l'entraînement
        callbacks=[
            #EarlyStopping(monitor='train_loss_epoch', patience=100, mode='min', verbose=True),
            ModelCheckpoint( dirpath=config.SAVE_DIR,filename='best_model',monitor='val_loss', mode='min', save_top_k=1)
        #fast_dev_run=True,  # Pour le debug rapide, à enlever pour l'entraînement complet
        #profiler= simple_profiler.SimpleProfiler()  # Pour le profiling, à enlever si pas besoin
        
           
        ]
    )

trainer.fit(model, train_loader, val_loader)


