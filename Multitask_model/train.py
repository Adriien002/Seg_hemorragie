# Import depuis data/
import data.dataset as dataset
import data.transform as T_mtsk

# Import depuis models/
from models.architecture import BasicUNetWithClassification
from models.lightning_module import MultiTaskHemorrhageModule

import utils
from monai.utils import set_determinism
from monai.data import DataLoader, PersistentDataset, CacheDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
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
pl.seed_everything(config_l["seed"], workers=True)
set_determinism(seed=config_l["seed"])
# Génération automatique de tags à partir de config
tags = [f"{k}:{v}" for k, v in config_l.items() if k in ["sharing_type", "optimizer", "model", "loss_weighting"]]

wandb_logger = WandbLogger(
    project="hemorrhage_multitask_test",
    group="noponderation",
    tags=tags,
    config=config_l,
    name="multitask_with_full_dataset_in_house"
)

torch.cuda.empty_cache()


data_seg_hemo = dataset.get_segmentation_data('train') # task: "seg_hemorragie"
data_seg_in_house = dataset.get_inhouse_segmentation_data('train') # task: "seg_nouveau_1"
data_classif = dataset.get_classification_data('train') # task: "classification"
data_cls_mask = [d for d in data_classif if d.get("has_mask", True)]
data_cls_nomask = [d for d in data_classif if not d.get("has_mask", True)]

transforms = T_mtsk.TaskBasedTransform_V3()

ds_seg_hemo = PersistentDataset(data_seg_hemo, transform=transforms.seg_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_seg_hemo"))
ds_seg_in_house = PersistentDataset(data_seg_in_house, transform=transforms.seg_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_seg_in_house"))

ds_cls_mask = PersistentDataset(data_cls_mask, transform=transforms.cls_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_cls_mask"))
ds_cls_nomask = PersistentDataset(data_cls_nomask, transform=transforms.cls_pipeline_no_mask, cache_dir=os.path.join(config.SAVE_DIR, "cache_cls_nomask"))

# 4. Fusionner le tout
train_dataset = ConcatDataset([
    ds_seg_hemo, ds_seg_in_house, ds_cls_mask, ds_cls_nomask
])

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=utils.multitask_collate_fn
)


val_transforms = T_mtsk.TaskBasedTransform_V3()


    
  

    
data_seg_hemo_val = dataset.get_segmentation_data('val') # task: "seg_hemorragie"
data_seg_in_house_val = dataset.get_inhouse_segmentation_data('val') # task: "seg_nouveau_1"
data_classif_val = dataset.get_classification_data('val') # task: "classification"
data_cls_mask_val = [d for d in data_classif_val if d.get("has_mask", True)]
data_cls_nomask_val = [d for d in data_classif_val if not d.get("has_mask", True)]


val_ds_seg_hemo = PersistentDataset(data_seg_hemo_val, transform=val_transforms.seg_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_seg_hemo_val"))
val_ds_seg_in_house = PersistentDataset(data_seg_in_house_val, transform=val_transforms.seg_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_seg_in_house_val"))
val_ds_cls_mask = PersistentDataset(data_cls_mask_val, transform=val_transforms.cls_pipeline, cache_dir=os.path.join(config.SAVE_DIR, "cache_cls_mask_val"))
val_ds_cls_nomask = PersistentDataset(data_cls_nomask_val, transform=val_transforms.cls_pipeline_no_mask, cache_dir=os.path.join(config.SAVE_DIR, "cache_cls_nomask_val"))

# 4. Fusionner le tout
val_dataset = ConcatDataset([
    val_ds_seg_hemo, val_ds_seg_in_house, val_ds_cls_mask, val_ds_cls_nomask
])

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    shuffle=False,  # Pas besoin de shuffle pour la validation
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=utils.multitask_collate_fn
)     

    # Modèle
model = MultiTaskHemorrhageModule(num_steps=len(train_loader) * config.num_epochs)
print(f"Total number of steps: {len(train_loader) * config.num_epochs}")
    
    # Trainer
trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=[1],
        default_root_dir=config.SAVE_DIR,
        logger= wandb_logger,
        #logger=TensorBoardLogger(SAVE_DIR, name="multitask_unet3
        #gradient_clip_val=1.0,  # Gradient clipping pour la stabilité
        log_every_n_steps=50,
        #accumulate_grad_batches=4 ,# Ajout car pour gérer les petites tailles de batch ( dues à limitation mémoire)
        #precision='16-mixed',  # Mixed precision pour accélérer l'entraînement
        callbacks=[
            #EarlyStopping(monitor='train_loss_epoch', patience=100, mode='min', verbose=True),
            ModelCheckpoint( dirpath=config.SAVE_DIR,filename='best_model',monitor='val_seg_loss', mode='min', save_top_k=1)
        #fast_dev_run=True,  # Pour le debug rapide, à enlever pour l'entraînement complet
        #profiler= simple_profiler.SimpleProfiler()  # Pour le profiling, à enlever si pas besoin
        
           
        ]
    )
    
    # Entraînement
trainer.fit(model, train_loader, val_loader)
    
print("Training completed!")
