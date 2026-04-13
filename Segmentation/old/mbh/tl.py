import os
import warnings
import pytorch_lightning as pl
import torch
from monai.data import DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from monai.networks.nets import UNet
import monai.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# ======================
# 1. CONFIGURATION
# ======================
DATASET_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI'
SAVE_DIR = "/home/tibia/Projet_Hemorragie/trleaning"
CHECKPOINT_PATH = "/home/tibia/Projet_Hemorragie/v1_log_test/lightning_logs/version_0/checkpoints/epoch=999-step=156000.ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 2. DATA PIPELINE (UNCHANGED)
# ======================
transforms = T.Compose([
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']),
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    T.RandCropByPosNegLabeld(
        keys=['image', 'seg'],
        image_key='image',
        label_key='seg',
        pos=5.0,
        neg=1.0,
        spatial_size=(96, 96, 96),
        num_samples=2
    ),
    T.RandScaleIntensityd(keys=['image'], factors=0.02, prob=0.5),
    T.RandShiftIntensityd(keys=['image'], offsets=0.05, prob=0.5),
    T.RandRotate90d(keys=['image', 'seg'], prob=0.5, max_k=2, spatial_axes=(0, 1)),
    T.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=[0, 1])
])

val_transforms = T.Compose([
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']),
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
])

def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]

# ======================
# 3. LIGHTNING MODULE
# ======================
class HemorrhageModel(pl.LightningModule):
    def __init__(self, num_steps, checkpoint_path=None):
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
        
        if checkpoint_path:
            self.load_pretrained_weights(checkpoint_path)

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        old_state_dict = checkpoint["state_dict"]
        new_state_dict = self.state_dict()

        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"Number of epochs: {checkpoint.get('epoch', 'N/A')}")
        print(f"Number of steps: {checkpoint.get('global_step', 'N/A')}")
        print(f"Model state dict keys: {list(old_state_dict.keys())}")

        for name, param in old_state_dict.items():
            if name in new_state_dict:
                if param.shape == new_state_dict[name].shape:
                    new_state_dict[name].copy_(param)
                    print(f"Poids transférés pour la couche : {name}")
                else:
                    print(f"Poids non transférés pour {name} (dimensions incompatibles : {param.shape} vs {new_state_dict[name].shape})")
            else:
                print(f"Poids {name} non trouvé dans le nouveau modèle")

        self.load_state_dict(new_state_dict)
        print("Poids pré-entraînés chargés avec succès)")

     
           
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.dice_metric = DiceHelper(include_background=False, softmax=True, num_classes=6, reduction='none')

    def forward(self, x):
        return self.model(x)

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
        y_hat = sliding_window_inference(x, roi_size=(96, 96, 96), sw_batch_size=2, predictor=self.model)
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.num_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}
        }

# ======================
# 4. TRAINING SETUP
# ======================
def main():
    num_epochs = 2000
    train_files = get_data_files(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/seg")
    val_files = get_data_files(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/seg")
    test_files = get_data_files(f"{DATASET_DIR}/test/img", f"{DATASET_DIR}/test/seg")

    train_dataset = PersistentDataset(train_files, transform=transforms, cache_dir=os.path.join(SAVE_DIR, "cache_train"))
    val_dataset = PersistentDataset(val_files, transform=val_transforms, cache_dir=os.path.join(SAVE_DIR, "cache_val"))
    test_dataset = PersistentDataset(test_files, transform=val_transforms, cache_dir=os.path.join(SAVE_DIR, "cache_test"))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize model with pre-trained checkpoint
    model = HemorrhageModel(num_steps=len(train_loader) * num_epochs, checkpoint_path=CHECKPOINT_PATH)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[1],
        default_root_dir=SAVE_DIR,
        logger=TensorBoardLogger(save_dir=SAVE_DIR, name="lightning_logs"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=130, mode="min"),

        ]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()