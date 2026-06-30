import warnings


warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

import torch
import pytorch_lightning as pl
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset

import config
import data.dataset as dataset
from data.transform import get_transforms
from models.lightning_module import ClassificationModule, DenseNet121Module

torch.cuda.set_device(0)
pl.seed_everything(42, workers=True)
set_determinism(seed=42)

wandb_logger = WandbLogger(
    project="hemorrhage_classification_article",
    group="baseline_3d",
    name="densenet121_3d_classification",
)

torch.cuda.empty_cache()

local_cache = "/volatile/Work/tibia_cache_hemorragie_clssif"
import os; os.makedirs(local_cache, exist_ok=True)

# ── Données ──────────────────────────────────────────────────────────────────
train_cls = dataset.get_classification_data("train")
train_seg = dataset.get_seg_classification_data("train")
train_all = train_cls + train_seg

val_cls = dataset.get_classification_data("val")
val_seg = dataset.get_seg_classification_data("val")
val_all = val_cls + val_seg

train_mask   = [d for d in train_all if d["has_mask"]]
train_nomask = [d for d in train_all if not d["has_mask"]]
val_mask     = [d for d in val_all   if d["has_mask"]]
val_nomask   = [d for d in val_all   if not d["has_mask"]]

ds_train_mask   = PersistentDataset(train_mask,   get_transforms("train", True),  os.path.join(local_cache, "cls_train_mask"))
ds_train_nomask = PersistentDataset(train_nomask, get_transforms("train", False), os.path.join(local_cache, "cls_train_nomask"))
ds_val_mask     = PersistentDataset(val_mask,     get_transforms("val",   True),  os.path.join(local_cache, "cls_val_mask"))
ds_val_nomask   = PersistentDataset(val_nomask,   get_transforms("val",   False), os.path.join(local_cache, "cls_val_nomask"))

train_loader = DataLoader(
    ConcatDataset([ds_train_mask, ds_train_nomask]),
    batch_size=config.batch_size, shuffle=True,
    num_workers=4, persistent_workers=True, pin_memory=True,
)
val_loader = DataLoader(
    ConcatDataset([ds_val_mask, ds_val_nomask]),
    batch_size=1, shuffle=False,
    num_workers=4, persistent_workers=True, pin_memory=True,
)

# ── Modèle ───────────────────────────────────────────────────────────────────
num_steps = len(train_loader) * config.num_epochs
#model = ClassificationModule(num_steps=num_steps)
model = DenseNet121Module(num_steps=num_steps)
print(f"Total steps: {num_steps}")

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs=config.num_epochs,
    accelerator="auto",
    devices=[0],
    default_root_dir=config.SAVE_DIR,
    logger=wandb_logger,
    gradient_clip_val=1.0,
    log_every_n_steps=50,
    check_val_every_n_epoch=5,
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(
            dirpath=config.SAVE_DIR, filename="best_model",
            monitor="val_loss", mode="min", save_top_k=1, every_n_epochs=5,
        ),
    ],
)

trainer.fit(model, train_loader, val_loader)
print("Training completed!")
