import os
import warnings

import pytorch_lightning as pl
import torch
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, DiceHelper
from monai.networks.nets import UNet
import monai.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# ======================
# 1. CONFIGURATION
# ======================
DATASET_DIR = '/home/tibia/Projet_Hemorragie/mbh_seg/nii'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/MBH_2_log"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 2. DATA PIPELINE (same as your original)
# ======================
transforms = T.Compose([
    # Loading transforms
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'), # who cares about the background ?
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),  # make sure all images are the same orientation
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']), # to isotropic spacing
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),  # make sure we have at least 96 slices
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),  # clip images

    # Let's crop 2 patches per case using positive negative instead of class by class
    #T.RandCropByPosNegLabeld(
     #   keys=['image', 'seg'],
      #  image_key='image',
       # label_key='seg',
        #pos=5.0,
        #neg=1.0,
        #spatial_size=(96, 96, 96),
        #num_samples=2
    #),
    T.RandCropByLabelClassesd(
    keys=["image", "seg"],
    label_key="seg",
    spatial_size=(96, 96, 96),  
    num_classes=6,
    ratios=[0.1, 0.3, 0.1, 0.1, 0.1, 0.1],  # + de poids pour les classes rares (classe 1 ici à 0.3 par ex)
    num_samples=4, 
    ),

    # Data augmentations
    # For intensity augmentations, small random transforms but often
    # For spatial augmentations, only along the sagittal and coronal axis
    T.RandScaleIntensityd(
        keys=['image'],
        factors=0.02,
        prob=0.5
    ),
    T.RandShiftIntensityd(
       keys=['image'],
        offsets=0.05,
        prob=0.5
    ),
    T.RandRotate90d(
        keys=['image', 'seg'],
        prob=0.5,
        max_k=2,
        spatial_axes=(0, 1)
    ),
    T.RandFlipd(
        keys=['image', 'seg'],
        prob=0.5,
        spatial_axis=[0, 1]
    )
])

val_transforms = T.Compose([
    # Loading transforms
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'), # who cares about the background ?
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),  # make sure all images are the same orientation
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']), # to isotropic spacing
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),  # make sure we have at least 96 slices
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),  # clip images
])


def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]


# ======================
# 3. LIGHTNING MODULE 
# ======================
class HemorrhageModel(pl.LightningModule):
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
        )
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True) # don't need to weight the dice ce loss
        self.dice_metric = DiceHelper(include_background=False,
                                      softmax=True,
                                      num_classes=6,
                                      reduction='none')

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]
        y_logits = self.model(x)

        loss = self.loss_fn(y_logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]

        y_hat = sliding_window_inference(x,
                                         roi_size=(96, 96, 96),
                                         sw_batch_size=2,
                                         predictor=self.model)

        # Loss
        loss = self.loss_fn(y_hat, y)

        scores, _ = self.dice_metric(y_hat, y)

        y_labels = y.unique().long().tolist()[1:]
        scores = {label: scores[0][label - 1].item() for label in y_labels}

        metrics = {f'dice_c{label}': score for label, score in scores.items()}
        metrics['val_loss'] = loss

        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):

        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=self.num_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": 'step'
            }
        }


# ======================
# 4. TRAINING SETUP
# ======================


def main():
    num_epochs = 1000
    # Load data (same as original)
    train_files = get_data_files(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/seg")
    val_files = get_data_files(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/seg")

    train_dataset = PersistentDataset(
        train_files,
        transform=transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_train")
    )

    val_dataset = PersistentDataset(
        val_files,
        transform=val_transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_val")
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize model with checkpoint if available
    model = HemorrhageModel(num_steps=len(train_loader) * num_epochs)
    print(f"Total number of steps : {len(train_loader) * num_epochs}")

    # Configure trainer with progress bar and checkpointing
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[1],
        default_root_dir=SAVE_DIR,
        logger=TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="lightning_logs"  # Dossier où sont stockés les logs
        )
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()