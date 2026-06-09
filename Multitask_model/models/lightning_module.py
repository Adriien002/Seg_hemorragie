import models.architecture as Arch
import os
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from torch.optim import SGD
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import config
import torch.nn as nn
import data.dataset as dataset

class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 0.5, mta_weight: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.num_steps = num_steps
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.mta_weight = mta_weight
        
        # Modèle multi-tâche
        self.model = Arch.BasicUNetWithClassification(
            spatial_dims=3,
            in_channels=1,
            out_channels_orig=6, # pour segmentation mbh
            out_channels_inhouse=4, # pour segmentation in-house
            num_cls_classes=config.NUM_CLASSES  # pour classification
        )
        
        # --- CORRECTION DEVICE POUR LES POIDS ---
        # On enregistre les poids comme un buffer du modèle. 
        # PyTorch Lightning les mettra automatiquement sur le bon GPU.
        pos_weights = dataset.compute_pos_weights(split="train") # Calcule les poids à partir des données d'entraînement
        pos_weights = torch.clip(pos_weights, min=1.0, max=10.0) # On clip les poids pour éviter des valeurs extrêmes
        self.register_buffer("pos_weights", pos_weights)
        
        # Fonctions de perte
        self.seg_loss_fn_mbh = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_inhouse = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_instance = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        # Métriques de segmentation
        self.seg_orig_dice = DiceHelper(include_background=False, softmax=True, num_classes=6, reduction='none')
        self.seg_inhouse_dice = DiceHelper(include_background=False, softmax=True, num_classes=4, reduction='none')
        self.seg_instance_dice = DiceHelper(include_background=False, softmax=True, num_classes=2, reduction='none')
        
       
        
        # Métriques de classification
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
      
    def forward(self, x, task="seg_orig"):
        return self.model(x, task=task)

    @staticmethod
    def _lmta_loss(P, S):
        # P: [B, C, d, h, w] — proba spatiale bottleneck
        # S: [B, C, D, H, W] — proba spatiale full-res (softmax déjà appliqué)
        eps = 1e-8
        S_down = F.adaptive_avg_pool3d(S, output_size=P.shape[2:])
        S_down = S_down / (S_down.sum(dim=1, keepdim=True) + eps)
        kl_ps = (P * (P.clamp(min=eps).log() - S_down.clamp(min=eps).log())).sum(dim=1).mean()
        kl_sp = (S_down * (S_down.clamp(min=eps).log() - P.clamp(min=eps).log())).sum(dim=1).mean()
        return kl_ps + kl_sp

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        # 1. Classification
        if batch.get("classification") is not None:
            x, y = batch["classification"]["image"], batch["classification"]["class_label"]
            _, cls_logits, P, S = self.model(x)
            cls_loss = self.cls_loss_fn(cls_logits, y)
            total_loss += self.cls_weight * cls_loss
            self.log("train_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                lmta = self._lmta_loss(P, S.detach())
                total_loss += self.mta_weight * lmta
                self.log("train_lmta_cls", lmta, batch_size=x.shape[0], on_epoch=True)

        # 2. Segmentation MBH (masques seuls, pas de label classif)
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            seg_logits, _, P, S = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y)
            total_loss += self.seg_weight * seg_loss
            self.log("train_seg_orig_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                lmta = self._lmta_loss(P, S.detach())
                total_loss += self.mta_weight * lmta
                self.log("train_lmta_seg", lmta, batch_size=x.shape[0], on_epoch=True)

        # 3. Seg MBH + Classification (vrais masques ET vrais labels)
        if batch.get("seg_multi") is not None:
            x   = batch["seg_multi"]["image"]
            y_seg = batch["seg_multi"]["label"]
            y_cls = batch["seg_multi"]["class_label"]
            seg_logits, cls_logits, P, S = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y_seg)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            total_loss += self.seg_weight * seg_loss + self.cls_weight * cls_loss
            self.log("train_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("train_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)

        # 4. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            seg_logits, _, _, _ = self.model(x, task="seg_inhouse")
            seg_loss = self.seg_loss_fn_inhouse(seg_logits, y)
            total_loss += self.seg_weight * seg_loss
            self.log("train_seg_inhouse_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        # 6. Segmentation INSTANCE
        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            seg_logits, _, _, _ = self.model(x, task="seg_instance")
            seg_loss = self.seg_loss_fn_instance(seg_logits, y)
            total_loss += self.seg_weight * seg_loss
            self.log("train_seg_instance_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        total_loss = 0.0

        # 1. Classification
        if batch.get("classification") is not None:
            x, y = batch["classification"]["image"], batch["classification"]["class_label"]
            _, cls_logits, _, _ = self.model(x)
            loss = self.cls_loss_fn(cls_logits, y)

            y_pred = torch.sigmoid(cls_logits)
            self.cls_auc.update(y_pred, y.int())
            self.cls_mean_auc.update(y_pred, y.int())

            total_loss += self.cls_weight * loss
            self.log("val_cls_loss", loss, batch_size=x.shape[0], on_epoch=True)

        # 2. Segmentation MBH (masques seuls)
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img)[0]
            )
            loss = self.seg_loss_fn_mbh(y_hat, y)
            scores, _ = self.seg_orig_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            sc = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_mbh_c{l}': s for l, s in sc.items()}, on_epoch=True, prog_bar=True)
            total_loss += loss
            self.log("val_seg_loss", loss, batch_size=x.shape[0])

        # 3. Seg MBH + Classification (vrais masques ET vrais labels)
        if batch.get("seg_multi") is not None:
            x     = batch["seg_multi"]["image"]
            y_seg = batch["seg_multi"]["label"]
            y_cls = batch["seg_multi"]["class_label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img)[0]
            )
            seg_loss = self.seg_loss_fn_mbh(y_hat, y_seg)
            scores, _ = self.seg_orig_dice(y_hat, y_seg)
            y_labels = y_seg.unique().long().tolist()[1:]
            if y_labels:
                sc = {label: scores[0][label - 1].item() for label in y_labels}
                self.log_dict({f'dice_mbh_multi_c{l}': s for l, s in sc.items()}, on_epoch=True)
            _, cls_logits, _, _ = self.model(x)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            y_pred = torch.sigmoid(cls_logits)
            self.cls_auc.update(y_pred, y_cls.int())
            self.cls_mean_auc.update(y_pred, y_cls.int())
            total_loss += seg_loss + self.cls_weight * cls_loss
            self.log("val_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("val_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)

        # 5. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img, task="seg_inhouse")[0]
            )
            loss = self.seg_loss_fn_inhouse(y_hat, y)
            scores, _ = self.seg_inhouse_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_inhouse_c{l}': s for l, s in scores.items()}, on_epoch=True, prog_bar=True)
            total_loss += loss
            self.log("val_seg_inhouse_loss", loss, batch_size=x.shape[0])

        # 4. Segmentation INSTANCE
        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img, task="seg_instance")[0]
            )
            loss = self.seg_loss_fn_instance(y_hat, y)
            scores, _ = self.seg_instance_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_instance_c{l}': s for l, s in scores.items()}, on_epoch=True, prog_bar=True)
            total_loss += loss
            self.log("val_seg_instance_loss", loss, batch_size=x.shape[0])

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
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

            for i in range(config.NUM_CLASSES):
                self.log(f'val_auc_class_{i}', class_auc[i].item(), on_epoch=True)

            self.cls_auc.reset()
            self.cls_mean_auc.reset()
            self.cls_mean_precision.reset()
            self.cls_mean_recall.reset()
            
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
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