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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger 
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

# ======================
# CONFIGURATION
# ======================
SEG_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
CLASSIFICATION_DATA_DIR = '/home/tibia/Projet_Hemorragie/MBH_label_case'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_multitask_with_equilbrage"
os.makedirs(SAVE_DIR, exist_ok=True)


NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# ======================
# MULTI-TASK NETWORK
# ======================
# But est de partager l'encodeur entre les deux tâches, donc modifié le Unet de base de MONAI



from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.utils import ensure_tuple_rep

# Logger 

config = dict(
    sharing_type="hard",  # "soft" ou "fine_tune"
    model="BasicUNetWithClassification",
    loss_weighting="gradnorm",
    dataset_size="full",
    batch_size=2,
    learning_rate=1e-3,
    optimizer="sgd",
    alpha_gradnorm=1.5,
    seed=42
)


    
wandb_logger = WandbLogger(
project="hemorrhage_multitask_test",
group="gradnorm",
tags=["full_dataset", "hard", "sgd", "unet3d", "noponderation"],
config=config
)



class BasicUNetWithClassification(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 6,  # pour segmentation
        num_cls_classes: int = 6,  # pour classification
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # Encoder
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # Decoder
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        # Classification head → à partir du bottleneck x4
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(fea[4] * 4 * 4 * 4, 512),
            nn.LayerNorm(512), #nn.BatchNorm1d(512)
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256), #nn.BatchNorm1d(256)
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_cls_classes)
        )

    def forward(self, x: torch.Tensor):
        # Encoder
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        # Decoder (segmentation)
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        seg_logits = self.final_conv(u1)

        # Classification
        cls_logits = self.cls_head(x4)  # x4 est le bottleneck

        return seg_logits  , cls_logits
    




# ======================
# DATA PREPARATION
# ======================
def get_segmentation_data(split="train"):
    img_dir = Path(SEG_DIR) / split / "img"
    seg_dir = Path(SEG_DIR) / split / "seg"
    
    images = sorted(img_dir.glob("*.nii.gz"))
    labels = sorted(seg_dir.glob("*.nii.gz"))
    
    assert len(images) == len(labels), "Mismatch between image and label counts"

    data = []
    for img, lbl in zip(images, labels):
        data.append({
            "image": str(img),
            "label": str(lbl),
            "task": "segmentation"
        })
        
    return data


def get_classification_data(split="train"):
    csv_path = Path(CLASSIFICATION_DATA_DIR) / "splits" / f"{split}_split.csv"
    df = pd.read_csv(csv_path)
    nii_dir = Path(CLASSIFICATION_DATA_DIR)
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    data = []
    for _, row in df.iterrows():
        image_path = str(nii_dir / f"{row['patientID_studyID']}.nii.gz")
        label = np.array([row[col] for col in label_cols], dtype=np.float32)
        
        data.append({
            "image": image_path,
            "label": label,
            "task": "classification"
        })
    return data


def get_multitask_dataset(split="train"):
    seg_data = get_segmentation_data(split)
    cls_data = get_classification_data(split)
    return seg_data + cls_data # équilibrer les tâches pour le multitâche

def get_multitask_dataset_balanced(split="train"):
    seg_data = get_segmentation_data(split)
    cls_data = get_classification_data(split)

    # Échantillonnage équilibré
    min_len = min(len(seg_data), len(cls_data))
    combined = []

    for i in range(min_len):
        combined.append(seg_data[i])
        combined.append(cls_data[i])

    # Ajouter les restants (si inégal)
    combined += seg_data[min_len:] + cls_data[min_len:]

    return combined

# ======================
# DATA TRANSFORMS
# ======================
# Idée est de faire deux pipelines de transformation, une pour la segmentation et une pour la classification



class TaskBasedTransform:
    """
    Applique un pipeline différent selon la tâche : "segmentation" ou "classification".
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedTransform initialized")

        self.window_preset = {"window_center": 40, "window_width": 80}

        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=True),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'],
                image_key='image',
                label_key='label',
                pos=5.0,
                neg=1.0,
                spatial_size=(96, 96, 96),
                num_samples=2
            ),
            T.RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5)
        ])

        self.cls_pipeline = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144)), #bof ( trop petit) pour classification mais on test
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            T.ToTensord(keys=["image", "label"])
        ])
        
    def __call__(self, data):
        task = data["task"]
        if task == "segmentation":
            return self.seg_pipeline(data)
        elif task == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {task}")
        
# def get_multitask_transforms():
#     return TaskBasedTransform(keys=["image", "label"])
   

        
class TaskBasedValTransform:
    """
    Transformations de validation — une pipeline par tâche, sans augmentation aléatoire.
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedTransform initialized")
        self.window_preset = {"window_center": 40, "window_width": 80}
   # Loading transforms
  # Ensure we load both image and segmentation

 # who cares about the background ?
  # make sure all images are the same orientation
   # to isotropic spacing
 # make sure we have at least 96 slices

        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=False),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            )
        ])

        self.cls_pipeline = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.ToTensord(keys=["image", "label"])
        ])


    def __call__(self, data):
        print("Pipeline called for task:", data["task"])
        if data["task"] == "segmentation":
            return self.seg_pipeline(data)
        elif data["task"] == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {data['task']}")
        
# def get_multitask_val_transforms():

#     return TaskBasedValTransform(keys=["image", "label"])


# ======================
# DATA COLLATE FUNCTION pour le multi-tâche et bien loader les batchs
# ======================


from monai.data.utils import list_data_collate
from collections.abc import Iterable

def flatten(batch):
    for item in batch:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def multitask_collate_fn(batch):
    flat_batch = list(flatten(batch))  

    classification_batch = []
    segmentation_batch = []

    for item in flat_batch:
        if item["task"] == "classification":
            classification_batch.append(item)
        elif item["task"] == "segmentation":
            segmentation_batch.append(item)
           
        else:
            raise ValueError(f"Tâche inconnue : {item['task']}")

    result = {
        "classification": list_data_collate(classification_batch) if classification_batch else None,
        "segmentation": list_data_collate(segmentation_batch) if segmentation_batch else None
    }
    
    return result

# Gestion equilibrage des classes
# TODO : ajouter un sur-échantillonnage pour la segmentation si nécessaire



# ======================
# LIGHTNING MODULE
# ======================
        
class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Pour GradNorm
        self.num_steps = num_steps
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.alpha = 1.5  # paramètre d’asymétrie GradNorm
        self.task_num = 2

        # Poids de pertes trainables
        self.w_seg = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.w_cls = nn.Parameter(torch.tensor(1.0, requires_grad=True))


        self.initial_losses = {}
   
        
        # Modèle multi-tâche
        self.model = BasicUNetWithClassification(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,  # pour segmentation
            num_cls_classes=NUM_CLASSES  # pour classification
        )
        
        # Fonctions de perte
        self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn = self._get_class_lossfn()  # BCEWithLogitsLoss avec poids de classe
        
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
        
        # Optimizers
        
        self.optimiser_model = None  # Initialisé dans configure_optimizers
        self.optimiser_w = None  # Initialisé dans configure_optimizers
   
    
    def _get_class_lossfn(self):
        pos_weights = torch.tensor([1.0] * NUM_CLASSES, dtype=torch.float)
        pos_weights = pos_weights.to(self.device)  

        print(f"Répartition des poids : {pos_weights}")
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    
      
    def forward(self, x):
        return self.model(x)

    
    def on_train_start(self):
        """Calcule et stocke les pertes initiales pour l'algorithme GradNorm."""
        
        
        self.model.eval()

        # Prend un seul lot du DataLoader de formation
        train_dataloader = self.trainer.train_dataloader
        batch = next(iter(train_dataloader))
       
        x_cls = batch["classification"]["image"] if batch["classification"] is not None else None
        y_cls = batch["classification"]["label"] if batch["classification"] is not None else None
        
        x_seg = batch["segmentation"]["image"] if batch["segmentation"] is not None else None
        y_seg = batch["segmentation"]["label"] if batch["segmentation"] is not None else None

        
        if x_cls is not None:
            x_cls, y_cls = x_cls.to(self.device), y_cls.to(self.device)
            _ , cls_logits = self.model(x_cls)
            initial_cls_loss = self.cls_loss_fn(cls_logits, y_cls).detach()
        else:
            initial_cls_loss = torch.tensor(0.0).to(self.device)
            
        if x_seg is not None:
            x_seg, y_seg = x_seg.to(self.device), y_seg.to(self.device)
            seg_logits, _ = self.model(x_seg)
            initial_seg_loss = self.seg_loss_fn(seg_logits, y_seg).detach()
        else:
            initial_seg_loss = torch.tensor(0.0).to(self.device)

        self.initial_losses = {
            "seg": initial_seg_loss,
            "cls": initial_cls_loss
        }
        
        print(f"Pertes initiales calculées: seg_loss={initial_seg_loss.item():.4f}, cls_loss={initial_cls_loss.item():.4f}")
        
        # Remet le modèle en mode d'entraînement
        self.model.train()


        
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        loss_cls = None
        loss_seg = None

        if batch["classification"] is not None:
            x_cls = batch["classification"]["image"]
            y_cls = batch["classification"]["label"]

        # Forward pass
            _ ,cls_logits = self.model(x_cls)

        # Loss classification
            loss_cls = self.cls_loss_fn(cls_logits, y_cls)
            
            total_loss+=self.w_cls*loss_cls
           

    #     # Log etc.

        if batch["segmentation"] is not None:
            x_seg = batch["segmentation"]["image"]
            y_seg = batch["segmentation"]["label"]

        # Forward pass
          
            seg_logits,_ = self.model(x_seg)

        # Loss segmentation
            loss_seg = self.seg_loss_fn(seg_logits, y_seg)
            total_loss += self.w_seg* loss_seg
            
        # 1. Choisir W = dernière couche partagée de l’encodeur
        shared_params = list(self.model.down_4.parameters())

            # 2. Calculer les normes de gradient pour chaque tâche
        if loss_seg is not None :
            G_seg = torch.autograd.grad(self.w_seg * loss_seg, shared_params, retain_graph=True, create_graph=True)
            G_seg_norm = torch.norm(torch.cat([g.reshape(-1) for g in G_seg]), 2)
        else:
            G_seg_norm = torch.tensor(0.0, device=self.device)
            
        if loss_cls is not None:
            G_cls = torch.autograd.grad(self.w_cls * loss_cls, shared_params, retain_graph=True, create_graph=True)
            G_cls_norm = torch.norm(torch.cat([g.reshape(-1) for g in G_cls]), 2)
        else:
            G_cls_norm = torch.tensor(0.0, device=self.device)

            # 3. Moyenne des normes
        G_avg = (G_seg_norm + G_cls_norm) / self.task_num

            # 4. Taux relatifs r_i(t)
        loss_seg_ratio = (loss_seg / self.initial_losses["seg"]).detach()
        loss_cls_ratio = (loss_cls / self.initial_losses["cls"]).detach()

        r_seg = loss_seg_ratio / ((loss_seg_ratio + loss_cls_ratio) / 2)
        r_cls = loss_cls_ratio / ((loss_seg_ratio + loss_cls_ratio) / 2)

            # 5. Cibles de normes
        target_seg = G_avg * (r_seg ** self.alpha)
        target_cls = G_avg * (r_cls ** self.alpha)

            # 6. Perte GradNorm
        L_grad = torch.abs(G_seg_norm - target_seg) + torch.abs(G_cls_norm - target_cls)
            
            # 7. Mise à jour des poids
        self.optimiser_w.zero_grad()
        self.manual_backward(L_grad, retain_graph=True)
        self.optimiser_w.step()
            
            
       


    # Backward sur perte totale (poids fixes à cette étape)
        self.optimiser_model.zero_grad()
        self.manual_backward(total_loss)
        self.optimiser_model.step()
        
        with torch.no_grad():
            sum_w = self.w_seg + self.w_cls
            self.w_seg *= self.task_num / sum_w
            self.w_cls *= self.task_num / sum_w
        # Log à compléter
        # Batch size pour log
        batch_size = 0
        if batch["classification"] is not None:
            batch_size += batch["classification"]["image"].shape[0]
        if batch["segmentation"] is not None:
            batch_size += batch["segmentation"]["image"].shape[0]

        self.log("train_loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        if loss_seg is not None:
            self.log("train_seg_loss", loss_seg, batch_size=batch_size, on_step=True, on_epoch=True)

        if loss_cls is not None:
            self.log("train_cls_loss", loss_cls, batch_size=batch_size, on_step=True, on_epoch=True)
        
        self.log("w_seg", self.w_seg, prog_bar=True)
        self.log("w_cls", self.w_cls, prog_bar=True)
      

        return total_loss
    
        
    def validation_step(self, batch,batch_idx):
        
        total_loss = 0.0
        loss_cls = None
        loss_seg = None
        
        if batch["classification"] is  not None : 
                x_cls = batch["classification"]["image"]
                y_cls = batch["classification"]["label"]
                
                _ , y_hat_cls = self.model(x_cls)
                loss_cls = self.cls_loss_fn(y_hat_cls, y_cls)
                y_cls_pred = torch.sigmoid(y_hat_cls).as_tensor()
                self.cls_auc.update(y_cls_pred, y_cls.int())
                self.cls_mean_auc.update(y_cls_pred, y_cls.int())
                self.cls_mean_precision.update(y_cls_pred, y_cls.int())
                self.cls_mean_recall.update(y_cls_pred, y_cls.int())
                
                total_loss += loss_cls* self.w_cls
                
        if batch["segmentation"] is not None:
                x_seg = batch["segmentation"]["image"]
                y_seg = batch["segmentation"]["label"]
                
                y_hat_seg = sliding_window_inference(
                    x_seg,
                    roi_size=(96, 96, 96),
                    sw_batch_size=2,
                    predictor=lambda x: self.model(x)[0]
                )
                
                loss_seg = self.seg_loss_fn(y_hat_seg, y_seg)
                scores, _ = self.seg_dice_metric(y_hat_seg, y_seg)
               
                y_labels = y_seg.unique().long().tolist()[1:]
                scores = {label: scores[0][label - 1].item() for label in y_labels}

                metrics = {f'dice_c{label}': score for label, score in scores.items()}
                

                self.log_dict(metrics, on_epoch=True, prog_bar=True)
                
                total_loss += loss_seg* self.w_seg

                
                # Log total loss
        batch_size = 0
        if batch["classification"] is not None:
                batch_size += batch["classification"]["image"].shape[0]
        if batch["segmentation"] is not None:
                batch_size += batch["segmentation"]["image"].shape[0]
                
      

        self.log("val_loss", total_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True )    

    


                
    
    def on_validation_epoch_end(self):
    # === CLASSIFICATION ===
        if len(self.cls_auc.preds) > 0:
            class_auc = self.cls_auc.compute()
            mean_auc = self.cls_mean_auc.compute()
            mean_precision = self.cls_mean_precision.compute()
            mean_recall = self.cls_mean_recall.compute()

            self.log_dict({
                'val_mean_auc': mean_auc,
                'val_mean_precision': mean_precision,
                'val_mean_recall': mean_recall
            }, on_epoch=True, prog_bar=True)

            for i in range(NUM_CLASSES):
                self.log(f'val_auc_class_{i}', class_auc[i].item(), on_epoch=True)

            self.cls_auc.reset()
            self.cls_mean_auc.reset()
            self.cls_mean_precision.reset()
            self.cls_mean_recall.reset()
        
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=5,
    #     verbose=True
    # )

    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": scheduler,
    #         "monitor": "val_loss",
    #         "interval": "epoch",
    #         "frequency": 1
    #     }
    # }
          
    def configure_optimizers(self):
        optimizer_model = torch.optim.SGD(
            [p for n, p in self.named_parameters() if not n.startswith("w_")],
            lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003
        )
        optimizer_w = torch.optim.Adam([self.w_seg, self.w_cls], lr=0.025)

        return [optimizer_model, optimizer_w]

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

# ======================
# TRAINING SETUP
# ======================
def main():
    num_epochs = 1000
    batch_size = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # pos_weights = calculate_pos_weights(CLASSIFICATION_DATA_DIR, CLASS_NAMES)
    # print(f"Pos weights: {dict(zip(CLASS_NAMES, pos_weights.tolist()))}")
    
    # Préparation des données
    train_data = get_multitask_dataset("train")
    val_data = get_multitask_dataset("val")
    # train_data=get_multitask_dataset_balanced('train')
    # val_data=get_multitask_dataset_balanced('val')
    # train_data= get_segmentation_data("train") 
    # val_data = get_segmentation_data("val")

    # print(f"Training samples: {len(train_data)}")
    # print(f"Validation samples: {len(val_data)}")
    
    # Transforms
    train_transforms, val_transforms = TaskBasedTransform(keys=["image", "label"]), TaskBasedValTransform(keys=["image", "label"])
    
   
    # Datasets
    train_dataset = PersistentDataset(
        train_data, #train_data,
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
        persistent_workers=True,
        collate_fn=multitask_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # ou 2 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
       collate_fn=multitask_collate_fn
    )
    
    # Modèle
    model = MultiTaskHemorrhageModule(num_steps=len(train_loader) * num_epochs)
    print(f"Total number of steps: {len(train_loader) * num_epochs}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=[0],
        default_root_dir=SAVE_DIR,
        logger=wandb_logger,
   
        #gradient_clip_val=1.0,  # Gradient clipping pour la stabilité
        log_every_n_steps=50,
        #accumulate_grad_batches=4 ,# Ajout car pour gérer les petites tailles de batch ( dues à limitation mémoire)
        precision='16-mixed',  # Mixed precision pour accélérer l'entraînement
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=True),
            ModelCheckpoint( dirpath=SAVE_DIR,filename='best_model',monitor='val_loss', mode='min', save_top_k=2)
        #fast_dev_run=True,  # Pour le debug rapide, à enlever pour l'entraînement complet
        #profiler= simple_profiler.SimpleProfiler()  # Pour le profiling, à enlever si pas besoin
        
           
        ]
    )
    
    # Entraînement
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
    
    