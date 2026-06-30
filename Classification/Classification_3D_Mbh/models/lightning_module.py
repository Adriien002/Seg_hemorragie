import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torchmetrics.classification import MultilabelAUROC, MultilabelPrecision, MultilabelRecall

import config
import data.dataset as dataset
from models.architecture import BasicUNetClassifier, DenseNet121Classifier


class ClassificationModule(pl.LightningModule):
    def __init__(self, num_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.num_steps = num_steps

        self.model = BasicUNetClassifier(
            spatial_dims=3,
            in_channels=1,
            num_cls_classes=config.NUM_CLASSES,
        )

        pos_weights = dataset.compute_pos_weights(split="train")
        pos_weights = torch.clip(pos_weights, min=1.0, max=10.0)
        self.register_buffer("pos_weights", pos_weights)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        self.cls_auc            = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc       = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall    = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].as_tensor(), batch["class_label"].as_tensor()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"].as_tensor(), batch["class_label"].as_tensor()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.sigmoid(logits)
        self.cls_auc.update(y_pred, y.int())
        self.cls_mean_auc.update(y_pred, y.int())
        self.cls_mean_precision.update(y_pred, y.int())
        self.cls_mean_recall.update(y_pred, y.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self):
        class_auc = self.cls_auc.compute()
        self.log_dict({
            "val_mean_auc":       self.cls_mean_auc.compute(),
            "val_mean_precision": self.cls_mean_precision.compute(),
            "val_mean_recall":    self.cls_mean_recall.compute(),
        }, on_epoch=True, prog_bar=True)
        for i, name in enumerate(config.CLASS_NAMES):
            self.log(f"val_auc_{name}", class_auc[i].item(), on_epoch=True)
        self.cls_auc.reset()
        self.cls_mean_auc.reset()
        self.cls_mean_precision.reset()
        self.cls_mean_recall.reset()

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=1e-3, momentum=0.99, weight_decay=0.00003, nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(0.0, float(self.num_steps - step) / float(max(1, self.num_steps))),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"},
        }


class DenseNet121Module(pl.LightningModule):
    def __init__(self, num_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.num_steps = num_steps

        self.model = DenseNet121Classifier(
            spatial_dims=3,
            in_channels=1,
            num_cls_classes=config.NUM_CLASSES,
        )

        pos_weights = dataset.compute_pos_weights(split="train")
        pos_weights = torch.clip(pos_weights, min=1.0, max=10.0)
        self.register_buffer("pos_weights", pos_weights)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        self.cls_auc            = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc       = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall    = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].as_tensor(), batch["class_label"].as_tensor()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"].as_tensor(), batch["class_label"].as_tensor()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        y_pred = torch.sigmoid(logits)
        self.cls_auc.update(y_pred, y.int())
        self.cls_mean_auc.update(y_pred, y.int())
        self.cls_mean_precision.update(y_pred, y.int())
        self.cls_mean_recall.update(y_pred, y.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self):
        class_auc = self.cls_auc.compute()
        self.log_dict({
            "val_mean_auc":       self.cls_mean_auc.compute(),
            "val_mean_precision": self.cls_mean_precision.compute(),
            "val_mean_recall":    self.cls_mean_recall.compute(),
        }, on_epoch=True, prog_bar=True)
        for i, name in enumerate(config.CLASS_NAMES):
            self.log(f"val_auc_{name}", class_auc[i].item(), on_epoch=True)
        self.cls_auc.reset()
        self.cls_mean_auc.reset()
        self.cls_mean_precision.reset()
        self.cls_mean_recall.reset()

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=1e-3, momentum=0.99, weight_decay=0.00003, nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(0.0, float(self.num_steps - step) / float(max(1, self.num_steps))),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"},
        }
