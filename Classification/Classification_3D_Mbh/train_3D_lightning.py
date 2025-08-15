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
from torch.optim import SGD,Adam
from transformers import get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

# === Hyperparams ===
NUM_CLASSES = 6
BATCH_SIZE = 8
EPOCHS = 1000
LR = 1e-3
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_3D_Classif_cosine"

class ClassifierModule(pl.LightningModule):
    def __init__(self, config = None, pos_weights=None , num_steps = None):
        super().__init__()
        self.save_hyperparameters()
        self.num_steps = num_steps
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES

        # Dynamically get the model from torchvision.models
        self.model = self._get_model()

        # Get positional weights
        self.loss_fn = self._get_lossfn()
        self.pos_weights = pos_weights.to(DEVICE) if pos_weights is not None else torch.ones(NUM_CLASSES, device=DEVICE)
        
        # Set up per-class metrics
        self.val_auc = MultilabelAUROC(num_labels=self.num_classes, average=None)


        # Also keep track of mean metrics for convenience
        self.val_mean_auc = MultilabelAUROC(num_labels=self.num_classes)
        self.val_mean_precision = MultilabelPrecision(num_labels=self.num_classes, threshold=0.5)
        self.val_mean_recall = MultilabelRecall(num_labels=self.num_classes, threshold=0.5)

   
    def _get_model(self):
        return ResNet(
        block='basic',
        layers=[1, 1, 1, 1],        # Beaucoup moins de couches (vs [2,2,2,2])
        block_inplanes=[32, 64, 128, 256],  # Moins de channels (vs [64,128,256,512])
        spatial_dims=3,
        n_input_channels=1,
        num_classes=NUM_CLASSES,
        conv1_t_size=7,
        conv1_t_stride=(2, 2, 2)    # Stride dans les 3 dimensions
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
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
        
        
    # def configure_optimizers(self):
   

    # # === Optimizer ===
    #     optimizer_name = "Adam"
    #     lr = LR

    #     if optimizer_name == "Adam":
    #         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    #     else:
    #         raise NotImplementedError(f"Optimiseur {optimizer_name} non supporté.")

    # # === Scheduler ===
    #     scheduler = ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',       
    #         factor=0.1,
    #         patience=3,
    #         verbose=False
    # )

    # # === Lightning format ===
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_loss", 
    #             "interval": "epoch",
    #             "frequency": 1
    #     }
    # }
    # def configure_optimizers(self):

    #     optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003)
    #     scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=0,
    #                                                 num_training_steps=self.num_steps)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "frequency": 1,
    #             "interval": 'step'
    #         }
    #      }
    def configure_optimizers(model):
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }
    }

    
# def calculate_pos_weights(csv_path, label_cols):
#     """Calculate pos_weight for BCEWithLogitsLoss based on class frequency."""
#     df = pd.read_csv(csv_path)
#     pos_weights = []
#     for col in label_cols:
#         pos_count = df[col].sum()
#         neg_count = len(df) - pos_count
#         if pos_count > 0:
#             pos_weight = neg_count / pos_count
#         else:
#             pos_weight = 1.0  # Default weight if no positive samples
#         pos_weights.append(pos_weight)
#     return torch.tensor(pos_weights, dtype=torch.float)

def main():
    # ---- Config ----
    csv_path = Path("/home/tibia/Projet_Hemorragie/MBH_label_case/splits/train_split.csv")
    nii_dir = Path("/home/tibia/Projet_Hemorragie/MBH_label_case")
    cache_dir = Path("./persistent_cache/3D_train_cache")  
    cache_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    df = pd.read_csv(csv_path)

# ---- Build MONAI-style data list ----
    data_list = [
        {
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32)
    }
        for _, row in df.iterrows()
        
]

    # pos_weights = calculate_pos_weights(csv_path, label_cols)
    # print(f"Pos weights: {dict(zip(label_cols, pos_weights.tolist()))}")

    train_transforms = T.Compose([
    # Load image only
        T.LoadImaged(keys=["image"], image_only=True),  
        T.EnsureChannelFirstd(keys=["image"]),
    
    # Harmonisation spatiale
        T.Orientationd(keys=["image"], axcodes='RAS'),
        T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    
   
        T.ResizeWithPadOrCropd(
            keys=["image"], 
            spatial_size=(224, 224, 144),
            mode="constant",  # Padding avec des zéros
            constant_values=0
    ),
    
    # Intensity normalization
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=-10,
            a_max=140,
            b_min=0.0,
            b_max=1.0,
            clip=True
    ),

    # Augmentations
        T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
        T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
        T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

    # Final tensor
        T.ToTensord(keys=["image", "label"])
])

# ---- PersistentDataset ----
    train_dataset = PersistentDataset(
        data=data_list,
        transform=train_transforms,
        cache_dir=str(cache_dir),
)

    print(f"Dataset ready with {len(train_dataset)} samples and cached transforms at {cache_dir}")

# Test pour vérifier les tailles
    print("Vérification des tailles des premières images:")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Image {i}: {sample['image'].shape}, Label: {sample['label'].shape}")




    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
)

    print(f"Using device: {DEVICE}")
    print(f"Number of Batches in the dataset: {len(train_loader)}")

    val_transforms = T.Compose([
        T.LoadImaged(keys=["image"], image_only=True),  
        T.EnsureChannelFirstd(keys=["image"]),
        T.Orientationd(keys=["image"], axcodes='RAS'),
        T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144), mode="constant", constant_values=0),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=-10,
            a_max=140,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        T.ToTensord(keys=["image", "label"])
])

# === Validation dataset ===
    val_csv_path = Path("/home/tibia/Projet_Hemorragie/MBH_label_case/splits/val_split.csv")
    val_df = pd.read_csv(val_csv_path)

    val_data_list = [
        {
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32)
        }
        for _, row in val_df.iterrows()
]

    val_cache_dir = Path("./persistent_cache/3D_val_cache")
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = PersistentDataset(
        data=val_data_list,
        transform=val_transforms, 
        cache_dir=str(val_cache_dir),
)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
)
    print(f"Validation dataset ready with {len(val_dataset)} samples and cached transforms at {val_cache_dir}")

    model =ClassifierModule(num_steps=len(train_loader) * EPOCHS)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=[0],
        default_root_dir=SAVE_DIR,
        logger=TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="lightning_logs"  # Dossier où sont stockés les logs
        ),
        precision= '16-mixed',
        accumulate_grad_batches=4,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=100, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}")
        ]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
    
    
    