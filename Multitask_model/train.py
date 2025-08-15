# Import depuis data/
import data.dataset as dataset
import data.transform as T_mtsk

# Import depuis models/
from models.architecture import BasicUNetWithClassification
from models.lightning_module import MultiTaskHemorrhageModule, MultiTaskHemorrhageModule_homeo, MultiTaskHemorrhageModule_gradnorm

import utils

from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import config

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")
## LOGGER
config_l = dict(
    sharing_type="hard",  # "soft" ou "fine_tune"
    model="BasicUNetWithClassification",
    loss_weighting="homeo",
    dataset_size="small : positifs cases",
    batch_size=2,
    learning_rate=1e-3,
    optimizer="sgd",
    seed=42
)


    
wandb_logger = WandbLogger(
project="hemorrhage_multitask_test",
group="homeo",
tags=["small_dataset", "hard", "sgd", "unet3d", "noponderation"],
config=config_l
)



#pos_weights = calculate_pos_weights(CLASSIFICATION_DATA_DIR, CLASS_NAMES)
# print(f"Pos weights: {dict(zip(CLASS_NAMES, pos_weights.tolist()))}")
    
# Préparation des données
# train_data = get_multitask_dataset("train")
# val_data = get_multitask_dataset("val")

train_data=dataset.get_balanced_multitask_dataset('train')
val_data=dataset.get_balanced_multitask_dataset('val')

# train_data= get_segmentation_data("train") 
# val_data = get_segmentation_data("val")


    
    # Transforms
train_transforms, val_transforms = T_mtsk.TaskBasedTransform(keys=["image", "label"]), T_mtsk.TaskBasedValTransform(keys=["image", "label"])
    
   
    # Datasets
train_dataset = PersistentDataset(
        train_data, 
        transform=train_transforms,
        cache_dir=os.path.join(config.SAVE_DIR, "cache_train")
    )
    
val_dataset = PersistentDataset(
        val_data,
        transform=val_transforms,
        cache_dir=os.path.join(config.SAVE_DIR, "cache_val")
    )
    
    # DataLoaders
train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        collate_fn=utils.multitask_collate_fn
    )
    
val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # ou 2 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
       collate_fn=utils.multitask_collate_fn
    )
    
    # Modèle
model = MultiTaskHemorrhageModule_homeo(num_steps=len(train_loader) * config.num_epochs)
print(f"Total number of steps: {len(train_loader) * config.num_epochs}")
    
    # Trainer
trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=[0],
        default_root_dir=config.SAVE_DIR,
        logger= wandb_logger,
        #logger=TensorBoardLogger(SAVE_DIR, name="multitask_unet3
        #gradient_clip_val=1.0,  # Gradient clipping pour la stabilité
        log_every_n_steps=50,
        accumulate_grad_batches=4 ,# Ajout car pour gérer les petites tailles de batch ( dues à limitation mémoire)
        precision='16-mixed',  # Mixed precision pour accélérer l'entraînement
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=True),
            ModelCheckpoint( dirpath=config.SAVE_DIR,filename='best_model',monitor='val_loss', mode='min', save_top_k=2)
        #fast_dev_run=True,  # Pour le debug rapide, à enlever pour l'entraînement complet
        #profiler= simple_profiler.SimpleProfiler()  # Pour le profiling, à enlever si pas besoin
        
           
        ]
    )
    
    # Entraînement
trainer.fit(model, train_loader, val_loader)
    
print("Training completed!")