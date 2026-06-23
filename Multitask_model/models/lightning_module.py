import models.architecture as Arch
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper, DiceMetric
from monai.networks.utils import one_hot
from torch.optim import SGD
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import data.dataset as dataset


class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(
        self,
        num_steps: int,
        cls_weight: float = 0.3,
        mta_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_steps = num_steps
        self.cls_weight = cls_weight
        self.mta_weight = mta_weight

        self.model = Arch.BasicUNetWithClassification(
            spatial_dims=3,
            in_channels=1,
            out_channels_orig=6,
            out_channels_inhouse=4,
            num_cls_classes=config.NUM_CLASSES,
            decoder_sharing=config.DECODER_SHARING,
        )

        pos_weights = dataset.compute_pos_weights(split="train")
        pos_weights = torch.clip(pos_weights, min=1.0, max=10.0)
        self.register_buffer("pos_weights", pos_weights)

        self.seg_loss_fn_mbh      = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_inhouse  = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.seg_loss_fn_instance = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn          = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        # UW (Uncertainty Weighting) — seg uniquement, indices : 0=MBH, 1=inhouse, 2=instance.
        # La classification garde un poids fixe (cls_weight) : la BCE peut tomber très bas
        # → σ²_cls → 0 → poids → ∞, instabilité irrécupérable.
        # s est clampé dans [-1.1, 0.5] → poids effectif 1/(2·exp(s)) ∈ [0.30, 1.50].
        self.uw_log_var = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5]))

        # Métriques par batch (dice par classe, pour wandb)
        self.seg_orig_dice     = DiceHelper(include_background=False, softmax=True, num_classes=6, reduction='none')
        self.seg_inhouse_dice  = DiceHelper(include_background=False, softmax=True, num_classes=4, reduction='none')
        self.seg_instance_dice = DiceHelper(include_background=False, softmax=True, num_classes=2, reduction='none')

        # Métriques agrégées sur l'epoch (pour sélection de checkpoint)
        self.val_dice = {
            "mbh":      DiceMetric(include_background=False, reduction="mean"),
            "inhouse":  DiceMetric(include_background=False, reduction="mean"),
            "instance": DiceMetric(include_background=False, reduction="mean"),
        }

        self.cls_auc            = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc       = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall    = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)

    def forward(self, x, task="seg_orig"):
        return self.model(x, task=task)



    def _uw(self, loss: torch.Tensor, seg_idx: int) -> torch.Tensor:
        """UW Kendall et al. avec clamp pour éviter qu'une tâche ne domine."""
        s = torch.clamp(self.uw_log_var[seg_idx], min=-1.1, max=0.5)
        return loss / (2.0 * s.exp()) + s / 2.0

    @staticmethod
    def _update_dice(metric, y_hat, y, num_classes):
        pred = torch.softmax(y_hat, dim=1).argmax(dim=1, keepdim=True)
        metric(one_hot(pred, num_classes, dim=1), one_hot(y, num_classes, dim=1))

    @staticmethod
    def _lmta_loss(P, S):
        eps = 1e-8
        S_down = F.adaptive_avg_pool3d(S, output_size=P.shape[2:])
        S_down = S_down / (S_down.sum(dim=1, keepdim=True) + eps)
        kl_ps = (P * (P.clamp(min=eps).log() - S_down.clamp(min=eps).log())).sum(dim=1).mean()
        kl_sp = (S_down * (S_down.clamp(min=eps).log() - P.clamp(min=eps).log())).sum(dim=1).mean()
        return kl_ps + kl_sp

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        # 1. Classification — poids fixe
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

        # 2. Segmentation MBH (masques seuls)
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            seg_logits, _, P, S = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y)
            total_loss += self._uw(seg_loss, 0)
            self.log("train_seg_orig_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            if self.mta_weight > 0:
                lmta = self._lmta_loss(P, S.detach())
                total_loss += self.mta_weight * lmta
                self.log("train_lmta_seg", lmta, batch_size=x.shape[0], on_epoch=True)

        # 3. Seg MBH + Classification (masques ET labels)
        if batch.get("seg_multi") is not None:
            x     = batch["seg_multi"]["image"]
            y_seg = batch["seg_multi"]["label"]
            y_cls = batch["seg_multi"]["class_label"]
            seg_logits, cls_logits, P, S = self.model(x)
            seg_loss = self.seg_loss_fn_mbh(seg_logits, y_seg)
            cls_loss = self.cls_loss_fn(cls_logits, y_cls)
            total_loss += self._uw(seg_loss, 0) + self.cls_weight * cls_loss
            self.log("train_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("train_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)

        # 4. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            seg_logits, _, _, _ = self.model(x, task="seg_inhouse")
            seg_loss = self.seg_loss_fn_inhouse(seg_logits, y)
            total_loss += self._uw(seg_loss, 1)
            self.log("train_seg_inhouse_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        # 5. Segmentation INSTANCE
        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            seg_logits, _, _, _ = self.model(x, task="seg_instance")
            seg_loss = self.seg_loss_fn_instance(seg_logits, y)
            total_loss += self._uw(seg_loss, 2)
            self.log("train_seg_instance_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            s_clamped = torch.clamp(self.uw_log_var, min=-1.1, max=0.5)
            eff_weights = 1.0 / (2.0 * s_clamped.exp())
        self.log_dict({
            "uw_seg_orig":     eff_weights[0],
            "uw_seg_inhouse":  eff_weights[1],
            "uw_seg_instance": eff_weights[2],
            "uw_cls":          self.cls_weight,
        }, on_epoch=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
            self._update_dice(self.val_dice["mbh"], y_hat, y, 6)
            scores, _ = self.seg_orig_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            sc = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_mbh_c{l}': s for l, s in sc.items()}, on_epoch=True, prog_bar=True)
            total_loss += self._uw(loss, 0)
            self.log("val_seg_loss", loss, batch_size=x.shape[0])

        # 3. Seg MBH + Classification
        if batch.get("seg_multi") is not None:
            x     = batch["seg_multi"]["image"]
            y_seg = batch["seg_multi"]["label"]
            y_cls = batch["seg_multi"]["class_label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img)[0]
            )
            seg_loss = self.seg_loss_fn_mbh(y_hat, y_seg)
            self._update_dice(self.val_dice["mbh"], y_hat, y_seg, 6)
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
            total_loss += self._uw(seg_loss, 0) + self.cls_weight * cls_loss
            self.log("val_seg_multi_seg_loss", seg_loss, batch_size=x.shape[0], on_epoch=True)
            self.log("val_seg_multi_cls_loss", cls_loss, batch_size=x.shape[0], on_epoch=True)

        # 4. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img, task="seg_inhouse")[0]
            )
            loss = self.seg_loss_fn_inhouse(y_hat, y)
            self._update_dice(self.val_dice["inhouse"], y_hat, y, 4)
            scores, _ = self.seg_inhouse_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_inhouse_c{l}': s for l, s in scores.items()}, on_epoch=True, prog_bar=True)
            total_loss += self._uw(loss, 1)
            self.log("val_seg_inhouse_loss", loss, batch_size=x.shape[0])

        # 5. Segmentation INSTANCE
        if batch.get("seg_instance") is not None:
            x, y = batch["seg_instance"]["image"], batch["seg_instance"]["label"]
            y_hat = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=2,
                predictor=lambda img: self.model(img, task="seg_instance")[0]
            )
            loss = self.seg_loss_fn_instance(y_hat, y)
            self._update_dice(self.val_dice["instance"], y_hat, y, 2)
            scores, _ = self.seg_instance_dice(y_hat, y)
            y_labels = y.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            self.log_dict({f'dice_instance_c{l}': s for l, s in scores.items()}, on_epoch=True, prog_bar=True)
            total_loss += self._uw(loss, 2)
            self.log("val_seg_instance_loss", loss, batch_size=x.shape[0])

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if len(self.cls_auc.preds) > 0:
            class_auc = self.cls_mean_auc.compute()
            self.log_dict({
                'val_mean_auc':       class_auc,
                'val_mean_precision': self.cls_mean_precision.compute(),
                'val_mean_recall':    self.cls_mean_recall.compute(),
            }, on_epoch=True, prog_bar=True)
            per_class = self.cls_auc.compute()
            for i in range(config.NUM_CLASSES):
                self.log(f'val_auc_class_{i}', per_class[i].item(), on_epoch=True)
            self.cls_auc.reset()
            self.cls_mean_auc.reset()
            self.cls_mean_precision.reset()
            self.cls_mean_recall.reset()

        dice_means = {}
        for name, metric in self.val_dice.items():
            try:
                agg = metric.aggregate()
                val = agg.item() if torch.is_tensor(agg) else float(agg)
            except ValueError:
                continue
            finally:
                metric.reset()
            if val == val:  # exclut NaN
                dice_means[name] = val
                self.log(f"val_dice_{name}", val, on_epoch=True, prog_bar=True)
        if dice_means:
            self.log("val_mean_dice_segds",
                     sum(dice_means.values()) / len(dice_means),
                     on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimiseur
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        uw_params  = [p for n, p in self.named_parameters() if "uw_log_var" in n]
        net_params = [p for n, p in self.named_parameters() if "uw_log_var" not in n]

        optimizer = SGD([
            {"params": net_params, "lr": 1e-3, "momentum": 0.99, "weight_decay": 0.00003, "nesterov": True},
            {"params": uw_params,  "lr": 1e-2, "momentum": 0.0,  "weight_decay": 0.0,     "nesterov": False},
        ])

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda step: max(0.0, float(self.num_steps - step) / float(max(1, self.num_steps))),
                lambda step: 1.0,  # LR constant pour uw_log_var
            ],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"},
        }


