import pprint
from typing import Any

import pandas as pd
import pytorch_lightning as pl
#import timm
from monai.networks.nets import ResNet
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, SequentialLR
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import monai.transforms as T
from monai.data import PersistentDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

# === Hyperparams ===
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_2D_Classif"




class ClassifierModule(pl.LightningModule):
    def __init__(self, config=None, pos_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        
        self.pos_weights = pos_weights.to(DEVICE) if pos_weights is not None else torch.ones(NUM_CLASSES, device=DEVICE)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        # Dynamically get the model from torchvision.models
        self.model = self._get_model()

        # Get positional weights
        self.loss_fn = self._get_lossfn()
        
        # Set up per-class metrics
        self.val_auc = MultilabelAUROC(num_labels=self.num_classes, average=None)


        # Also keep track of mean metrics for convenience
        self.val_mean_auc = MultilabelAUROC(num_labels=self.num_classes)
        self.val_mean_precision = MultilabelPrecision(num_labels=self.num_classes, threshold=0.5)
        self.val_mean_recall = MultilabelRecall(num_labels=self.num_classes, threshold=0.5)

   
    def _get_model(self):
        return ResNet(
        block='basic',           # BasicBlock for ResNet18/34
        layers=[2, 2, 2, 2],    # ResNet18 architecture
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=2,
        n_input_channels=1,     # Input = Scan
        num_classes=NUM_CLASSES,
        conv1_t_size=7,
        conv1_t_stride=2 
        
    
    ).to(DEVICE)

    def _get_lossfn(self):
        pos_weights = torch.tensor([1.0] * self.num_classes, dtype=torch.float)
        pos_weights = pos_weights.to(self.device)  

        print(f"Répartition des poids : {pos_weights}")
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        y_pred = torch.sigmoid(y_hat).as_tensor()

        # Update both per-class and mean metrics
        self.val_auc.update(y_pred, y.int())
     
        self.val_mean_auc.update(y_pred, y.int())
        self.val_mean_precision.update(y_pred, y.int())
        self.val_mean_recall.update(y_pred, y.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Calculate auc per class
        class_auc = self.val_auc.compute()
       
        # Calculate mean metrics
        mean_auc= self.val_mean_auc.compute()
        mean_specificity = self.val_mean_precision.compute()
        mean_recall = self.val_mean_recall.compute()

        # Log mean metrics for progress bar
        self.log_dict({
            'mean_auc': mean_auc,
            'mean_spe': mean_specificity,
            'mean_rec': mean_recall
        }, on_epoch=True)

        # Log per-class metrics
        metrics_dict = {}
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            metrics_dict.update({
                f'auc_{class_name}': class_auc[i].item()
            })

        self.log_dict(metrics_dict, on_epoch=True)

        # Reset all metrics
        self.val_auc.reset()
        
        self.val_mean_auc.reset()
        self.val_mean_precision.reset()
        self.val_mean_recall.reset()

    
    def configure_optimizers(self):
   

    # === Optimizer ===
        optimizer_name = "Adam"
        lr = LR

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimiseur {optimizer_name} non supporté.")

    # === Scheduler ===
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',       
            factor=0.1,
            patience=3,
            verbose=False
    )

    # === Lightning format ===
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_loss', 
                "interval": "epoch",
                "frequency": 1
        }
    }
        
        
# On reprends logiaue code sans lightning
def prepare_data(csv_path, dicom_dir, label_cols):

        df = pd.read_csv(csv_path)
    
   
        data_list = [
        {
            "image": str(dicom_dir / row.filename),
            "label": np.array([row[col] for col in label_cols], dtype=np.float32)
        }
        for _, row in df.iterrows() # itertuples est plus rapide que iterrows?
    ]
    
        return data_list



def create_transforms():
    window_preset = {"window_center": 40, "window_width": 80}
    train_transforms = T.Compose([
        T.LoadImaged(keys=["image"], image_only=True),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
            a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        T.EnsureChannelFirstd(keys=["image"]),
        T.Resized(keys=["image"], spatial_size=(256, 256)),
        T.RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Horizontal flip
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Vertical flip
        T.RandAffined(
            keys=["image"],
            prob=0.3,
            rotate_range=(0.1, 0.1),  # Slight rotation
            scale_range=(0.1, 0.1),   # Slight zoom
            mode='bilinear'
        ),
        T.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        T.RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
        T.ToTensord(keys=["image", "label"])
    ])
    
    val_transforms = T.Compose([
        T.LoadImaged(keys=["image"], image_only=True),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
            a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        T.EnsureChannelFirstd(keys=["image"]),
        T.Resized(keys=["image"], spatial_size=(256, 256)),
        T.ToTensord(keys=["image", "label"])
    ])
    
    return train_transforms, val_transforms

def calculate_pos_weights(csv_path, label_cols):
    """Calculate pos_weight for BCEWithLogitsLoss based on class frequency."""
    df = pd.read_csv(csv_path)
    pos_weights = []
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count
        if pos_count > 0:
            pos_weight = neg_count / pos_count
        else:
            pos_weight = 1.0  # Default weight if no positive samples
        pos_weights.append(pos_weight)
    return torch.tensor(pos_weights, dtype=torch.float)

def main():
    csv_train_path = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/csv/train_fold0.csv")
    csv_val_path = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/csv/val_fold0.csv")
    dicom_dir = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/rsna-intracranial-hemorrhage-detection/stage_2_train")
    train_cache_dir = Path("./persistent_cache2/fold0_train") 
    val_cache_dir = Path("./persistent_cache2/fold0_val")
    
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # === Pos Weights ===
    pos_weights = calculate_pos_weights(csv_train_path, label_cols)
    print(f"Pos weights: {dict(zip(label_cols, pos_weights.tolist()))}")
    
    # === Prepare Data ===
    print("Preparing data...")
    data_train_list = prepare_data(csv_train_path, dicom_dir, label_cols)
    data_val_list = prepare_data(csv_val_path, dicom_dir, label_cols)

    
    # === Create Transforms ===
    train_transforms,val_transforms= create_transforms()
    
    # === Create Dataset ===
    train_dataset = PersistentDataset(
    data=data_train_list,
    transform=train_transforms,
    cache_dir=str(train_cache_dir),
    )

    val_dataset = PersistentDataset(
    data=data_val_list,
    transform=val_transforms,
    cache_dir=str(val_cache_dir),
    )
    # train_dataset = Dataset(data=data_train_list, transform=train_transforms)
    # val_dataset = Dataset(data=data_val_list, transform=train_transforms)


    print(f"training dataset ready with {len(train_dataset)} samples and cached transforms at {train_cache_dir}")
    print(f"validation dataset ready with {len(val_dataset)} samples and cached transforms at {val_cache_dir}")
    
    # === Create DataLoader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )



    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"Number of Batches in the training dataset: {len(train_loader)}")
    print(f"Number of Batches in the validation dataset: {len(val_loader)}")



    model = ClassifierModule(pos_weights=pos_weights)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=[0],
        default_root_dir=SAVE_DIR,
        logger=TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="lightning_logs"  # Dossier où sont stockés les logs
        ),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}")
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    


if __name__ == "__main__":
    main()
    
    
    