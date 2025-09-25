

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from transformers import get_linear_schedule_with_warmup
import config
import os
import monai.networks.nets as monai_nets
import torch
import numpy as np

class HemorrhageModel(pl.LightningModule):
    def __init__(self, num_steps):
        super().__init__()
        self.config=config.CONFIG
        
        
        self.num_steps = num_steps
        self.model = monai_nets.UNet(**self.config["model"])
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

        metrics = {f'val_dice_c{label}': score for label, score in scores.items()}
        metrics['val_loss'] = loss

        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        return loss
    
    
    def predict_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]
        
        # Inférence avec sliding window
        y_hat = sliding_window_inference(
            x,
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=self.model
        )
        
        # Calcul des scores Dice
        scores, _ = self.dice_metric(y_hat, y)
        
        # Extraction des scores par classe
        dice_scores = {}
        for class_idx in range(5):  # Classes 1-5
            dice_scores[f"dice_c{class_idx+1}"] = scores[0, class_idx].item()
        
        if isinstance(batch["image"].meta["filename_or_obj"], list):
            full_path = batch["image"].meta["filename_or_obj"][0]
        else:
            full_path = batch["image"].meta["filename_or_obj"]
        
        filename = os.path.basename(full_path)
         
        
        return {
            'preds': y_hat.cpu(),  
            'dice': dice_scores,
            'filename': filename,
            'ground_truth': y.cpu(),
            'original_image': x.cpu(),
            'affine': batch["image"].meta.get("affine", torch.eye(4)),
            'original_shape': batch["image"].meta.get("spatial_shape", y_hat.shape[2:]),
            'image_meta_dict': batch["image"].meta                                        
        }
  
        
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]

        y_hat = sliding_window_inference(x,
                                         roi_size=(96, 96, 96),
                                         sw_batch_size=2,
                                         predictor=self.model)

        # Loss
        loss = self.loss_fn(y_hat, y)

        scores, _ = self.dice_metric(y_hat, y)

        y_labels = y.unique().long().tolist()[1:] # exclude background
        scores = {label: scores[0][label - 1].item() for label in y_labels} # maps label to its dice score

        metrics = {f'test_dice_c{label}': score for label, score in scores.items()}
        metrics['test_loss'] = loss

        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        return loss

        
    def configure_optimizers(self):

        cfg = self.config

    # Choix de l'optimizer
        if cfg["training"]["optimizer"] == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"]
            )
        elif cfg["training"]["optimizer"] == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=cfg["training"]["learning_rate"],
                momentum=cfg["training"]["momentum"],
                nesterov=True,
                weight_decay=cfg["training"]["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer {cfg['training']['optimizer']} not supported.")

        # Scheduler
        if cfg["scheduler"]["type"] == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg["scheduler"]["num_warmup_steps"],
                num_training_steps=self.num_steps
            )
            
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": 'step'
            }
        }
        
    # def configure_optimizers(model):
    #     optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": scheduler,
    #         "interval": "epoch",
    #         "frequency": 1
    #     }
    # }


