import os
import warnings
from typing import Dict, Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from monai.networks.nets import UNet
import monai.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# ======================
# CONFIGURATION
# ======================
SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_log"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# ======================
# MULTI-TASK NETWORK
# ======================
class MultiTaskHemorrhageNet(nn.Module):
    def __init__(self, num_seg_classes=6, num_cls_classes=6):
        super().__init__()
        
        # Encodeur partagé basé sur UNet
        self.shared_encoder = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=32,  # Features intermédiaires
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            up_kernel_size=3,
            act=('LeakyReLU', {'inplace': True}),
        )
        
        # Tête de segmentation
        self.seg_head = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, num_seg_classes, kernel_size=1)
        )
        
        # Tête de classification
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # Global pooling adaptatif
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_cls_classes)
        )
        
    def forward(self, x):
        # Encodage partagé
        shared_features = self.shared_encoder(x)
        
        # Segmentation
        seg_logits = self.seg_head(shared_features)
        
        # Classification
        cls_logits = self.cls_head(shared_features)
        
        return seg_logits, cls_logits

# ======================
# DATA TRANSFORMS
# ======================
def get_multitask_transforms():
    """Transforms pour les données multi-tâche"""
    window_preset = {"window_center": 40, "window_width": 80}
    
    train_transforms = T.Compose([
        # Loading transforms
        T.LoadImaged(keys=["image", "seg"]),
        T.EnsureChannelFirstd(keys=["image", "seg"]),
        T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
        T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
        T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']),
        T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
        
        # Intensity normalization pour segmentation ET classification
        T.ScaleIntensityRanged(
            keys=["image"], 
            a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
            a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        
        # Cropping pour segmentation
        T.RandCropByPosNegLabeld(
            keys=['image', 'seg'],
            image_key='image',
            label_key='seg',
            pos=5.0,
            neg=1.0,
            spatial_size=(96, 96, 64),
            num_samples=2
        ),
        
        # Augmentations
        T.RandScaleIntensityd(keys=['image'], factors=0.02, prob=0.5),
        T.RandShiftIntensityd(keys=['image'], offsets=0.05, prob=0.5),
        T.RandRotate90d(keys=['image', 'seg'], prob=0.5, max_k=2, spatial_axes=(0, 1)),
        T.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=[0, 1]),


       # T.ToTensord(keys=["image", "seg", "label"]) 
    ])
    
    
    val_transforms = T.Compose([
        T.LoadImaged(keys=["image", "seg"]),
        T.EnsureChannelFirstd(keys=["image", "seg"]),
        T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
        T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
        T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']),
        T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
            a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        #T.ToTensord(keys=["image", "seg", "label"]) 
    ])
    
    return train_transforms, val_transforms

# ======================
# DATA PREPARATION
# ======================
def prepare_multitask_data(seg_dir: str, cls_data_dir: str, split: str = "train"):
    """Prépare les données pour l'entraînement multi-tâche"""
    
    # Données de segmentation
    seg_img_dir = os.path.join(seg_dir, split, "img")
    seg_label_dir = os.path.join(seg_dir, split, "seg")
    
    seg_images = sorted([os.path.join(seg_img_dir, f) for f in os.listdir(seg_img_dir) if f.endswith('.nii.gz')])
    seg_labels = sorted([os.path.join(seg_label_dir, f) for f in os.listdir(seg_label_dir) if f.endswith('.nii.gz')])
    
    # Données de classification
    cls_csv_path = os.path.join(cls_data_dir, "splits", f"{split}_split.csv")
    cls_df = pd.read_csv(cls_csv_path)
    
    # Création du mapping pour les labels de classification
    cls_labels_dict = {}
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    for _, row in cls_df.iterrows():
        patient_id = row['patientID_studyID']
        cls_labels_dict[patient_id] = np.array([row[col] for col in label_cols], dtype=np.float32)
    
    # Combinaison des données
    multitask_data = []
    for img_path, seg_path in zip(seg_images, seg_labels):
        # Extraction de l'ID patient depuis le nom de fichier
        img_name = os.path.basename(img_path)
        patient_id = img_name.replace('.nii.gz', '')
        
        # Ajout des labels de classification si disponibles
        if patient_id in cls_labels_dict:
            multitask_data.append({
                "image": img_path,
                "seg": seg_path,
                "cls_labels": cls_labels_dict[patient_id]
            })
        else:
            # Si pas de labels de classification, utiliser des zéros
            multitask_data.append({
                "image": img_path,
                "seg": seg_path,
                "cls_labels": np.zeros(NUM_CLASSES, dtype=np.float32)
            })
    
    return multitask_data

# ======================
# LIGHTNING MODULE
# ======================
class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_steps = num_steps
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
        # Modèle multi-tâche
        self.model = MultiTaskHemorrhageNet(num_seg_classes=6, num_cls_classes=6)
        
        # Fonctions de perte
        self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        
        # Métriques de segmentation
        self.seg_dice_metric = DiceHelper(
            include_background=False,
            softmax=True,
            num_classes=6,
            reduction='none'
        )
        
        # Métriques de classification
        self.cls_auc = MultilabelAUROC(num_labels=NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x = batch["image"]
        seg_target = batch["seg"]
        cls_target = batch["cls_labels"]
        
        # Forward pass
        seg_logits, cls_logits = self.forward(x)
        
        # Calcul des pertes
        seg_loss = self.seg_loss_fn(seg_logits, seg_target)
        cls_loss = self.cls_loss_fn(cls_logits, cls_target)
        
        # Perte totale pondérée
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        # Logging
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True)
        
        # Learning rate
        if self.trainer.lr_scheduler_configs:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=True)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        seg_target = batch["seg"]
        cls_target = batch["cls_labels"]
        
        # Forward pass avec sliding window pour la segmentation
        seg_pred = sliding_window_inference(
            x, 
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=lambda x: self.forward(x)[0]  # Seulement la sortie de segmentation
        )
        
        # Forward pass normal pour la classification
        _, cls_logits = self.forward(x)
        
        # Calcul des pertes
        seg_loss = self.seg_loss_fn(seg_pred, seg_target)
        cls_loss = self.cls_loss_fn(cls_logits, cls_target)
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        # Métriques de segmentation
        seg_scores, _ = self.seg_dice_metric(seg_pred, seg_target)
        seg_labels = seg_target.unique().long().tolist()[1:]
        seg_metrics = {label: seg_scores[0][label - 1].item() for label in seg_labels}
        
        # Métriques de classification
        cls_pred = torch.sigmoid(cls_logits)
        self.cls_auc.update(cls_pred, cls_target.int())
        self.cls_mean_auc.update(cls_pred, cls_target.int())
        self.cls_mean_precision.update(cls_pred, cls_target.int())
        self.cls_mean_recall.update(cls_pred, cls_target.int())
        
        # Logging
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_seg_loss", seg_loss, on_epoch=True)
        self.log("val_cls_loss", cls_loss, on_epoch=True)
        
        # Log segmentation metrics
        for label, score in seg_metrics.items():
            self.log(f'val_dice_c{label}', score, on_epoch=True)
            
        return total_loss
        
    def on_validation_epoch_end(self):
        # Métriques de classification
        class_auc = self.cls_auc.compute()
        mean_auc = self.cls_mean_auc.compute()
        mean_precision = self.cls_mean_precision.compute()
        mean_recall = self.cls_mean_recall.compute()
        
        # Log des métriques moyennes
        self.log_dict({
            'val_cls_mean_auc': mean_auc,
            'val_cls_mean_precision': mean_precision,
            'val_cls_mean_recall': mean_recall
        }, on_epoch=True)
        
        # Log des métriques par classe
        for i, class_name in enumerate(CLASS_NAMES):
            self.log(f'val_cls_auc_{class_name}', class_auc[i].item(), on_epoch=True)
        
        # Reset des métriques
        self.cls_auc.reset()
        self.cls_mean_auc.reset()
        self.cls_mean_precision.reset()
        self.cls_mean_recall.reset()
        
    def configure_optimizers(self):
        # Optimiseur adaptatif selon les tâches
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

# ======================
# TRAINING SETUP
# ======================
def main():
    num_epochs = 100
    batch_size = 2
    
    # Préparation des données
    train_data = prepare_multitask_data(DATASET_DIR, CLASSIFICATION_DATA_DIR, "train")
    val_data = prepare_multitask_data(DATASET_DIR, CLASSIFICATION_DATA_DIR, "val")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Transforms
    train_transforms, val_transforms = get_multitask_transforms()
    
    # Datasets
    train_dataset = PersistentDataset(
        train_data,
        transform=train_transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_train")
    )
    
    val_dataset = PersistentDataset(
        val_data,
        transform=val_transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_val")
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True
    )
    
    # Modèle
    model = MultiTaskHemorrhageModule(
        num_steps=len(train_loader) * num_epochs,
        seg_weight=1.0,  # Poids pour la segmentation
        cls_weight=0.5   # Poids pour la classification
    )
    
    print(f"Total number of steps: {len(train_loader) * num_epochs}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[1],
        default_root_dir=SAVE_DIR,
        logger=TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="multitask_logs"
        ),
        gradient_clip_val=1.0,  # Gradient clipping pour la stabilité
        log_every_n_steps=50
        accumulate_grad_batches=4 # Ajout car pour gérer les petites tailles de batch ( dues à limitation mémoire)
    )
    
    # Entraînement
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()