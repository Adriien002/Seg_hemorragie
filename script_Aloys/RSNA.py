import pprint
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, SequentialLR
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelSpecificity
from transformers import get_cosine_schedule_with_warmup

from src.utils.loss import MultiLabelFocalLoss


class ClassifierModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = config["model"]["num_classes"]
        self.class_names = config.get("class_names", [f'class_{i}' for i in range(self.num_classes)])

        # Dynamically get the model from torchvision.models
        self.model = self._get_model(config["model"])

        # Get positional weights

        self.criterion = self._get_lossfn(config["train"])

        # Set up per-class metrics
        self.val_auc = MultilabelAUROC(num_labels=self.num_classes, average=None)
        self.val_specificity = MultilabelSpecificity(num_labels=self.num_classes, threshold=0.5, average=None)
        self.val_recall = MultilabelRecall(num_labels=self.num_classes, threshold=0.5, average=None)

        # Also keep track of mean metrics for convenience
        self.val_mean_auc = MultilabelAUROC(num_labels=self.num_classes)
        self.val_mean_specificity = MultilabelSpecificity(num_labels=self.num_classes, threshold=0.5)
        self.val_mean_recall = MultilabelRecall(num_labels=self.num_classes, threshold=0.5)

    @staticmethod
    def _get_model(model_config):
        model_name = model_config["name"]
        num_classes = model_config["num_classes"]
        pretrained = model_config.get("pretrained", False)

        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            print(f"Successfully loaded model '{model_name}' from timm.")
        except Exception as e_timm:
            print(f"Could not load model '{model_name}' from timm: {e_timm}")
            raise ValueError(f"Model {model_name} not found or could not be loaded from timm.")

        return model

    def _get_lossfn(self, trn_config):
        loss_name = trn_config.get('loss_name', 'bce')
        if loss_name == 'bce':
            pos_weights = torch.tensor(
                trn_config.get("pos_weights", [1.0] * self.num_classes),
                dtype=torch.float
            )
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        elif loss_name == 'focal':
            return MultiLabelFocalLoss()

        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        # Log the learning rate and training loss
        lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log('trn_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        y_pred = torch.sigmoid(y_hat).as_tensor()

        # Update both per-class and mean metrics
        self.val_auc.update(y_pred, y.int())
        self.val_specificity.update(y_pred, y.int())
        self.val_recall.update(y_pred, y.int())

        self.val_mean_auc.update(y_pred, y.int())
        self.val_mean_specificity.update(y_pred, y.int())
        self.val_mean_recall.update(y_pred, y.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Calculate per-class metrics
        class_accuracy = self.val_auc.compute()
        class_specificity = self.val_specificity.compute()
        class_recall = self.val_recall.compute()

        # Calculate mean metrics
        mean_accuracy = self.val_mean_auc.compute()
        mean_specificity = self.val_mean_specificity.compute()
        mean_recall = self.val_mean_recall.compute()

        # Log mean metrics for progress bar
        self.log_dict({
            'mean_auc': mean_accuracy,
            'mean_spe': mean_specificity,
            'mean_rec': mean_recall
        }, on_epoch=True)

        # Log per-class metrics
        metrics_dict = {}
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            metrics_dict.update({
                f'auc_{class_name}': class_accuracy[i].item(),
                f'spe_{class_name}': class_specificity[i].item(),
                f'rec_{class_name}': class_recall[i].item(),
            })

        self.log_dict(metrics_dict, on_epoch=True)

        # Reset all metrics
        self.val_auc.reset()
        self.val_specificity.reset()
        self.val_recall.reset()
        self.val_mean_auc.reset()
        self.val_mean_specificity.reset()
        self.val_mean_recall.reset()

    def on_test_start(self):
        self.prob_summary = {}

    def test_step(self, batch, batch_idx):
        fnames, x, y = batch['fname'], batch['image'], batch['label']
        y_hat = self.model(x)

        y_pred = torch.sigmoid(y_hat.as_tensor())

        for i, fname in enumerate(fnames):
            self.prob_summary[fname] = y_pred[i].cpu().tolist()

        # Update both per-class and mean metrics
        self.val_auc.update(y_pred, y.int())
        self.val_specificity.update(y_pred, y.int())
        self.val_recall.update(y_pred, y.int())

        self.val_mean_auc.update(y_pred, y.int())
        self.val_mean_specificity.update(y_pred, y.int())
        self.val_mean_recall.update(y_pred, y.int())

    def on_test_epoch_end(self):
        # Calculate per-class metrics
        class_accuracy = self.val_auc.compute()
        class_specificity = self.val_specificity.compute()
        class_recall = self.val_recall.compute()

        # Calculate mean metrics
        mean_auc = self.val_mean_auc.compute()
        mean_specificity = self.val_mean_specificity.compute()
        mean_recall = self.val_mean_recall.compute()

        # Log per-class metrics
        metrics_dict = {
            "mean_auc": mean_auc,
            "mean_spe": mean_specificity,
            "mean_rec": mean_recall
        }
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            metrics_dict.update({
                f'auc_{class_name}': class_accuracy[i].item(),
                f'spe_{class_name}': class_specificity[i].item(),
                f'rec_{class_name}': class_recall[i].item(),
            })

        print("#########")
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(metrics_dict)
        print("#########")

        df_prob_summary = pd.DataFrame.from_dict(self.prob_summary, orient='index', columns=self.class_names)
        df_prob_summary.index.name = 'fname'
        csv_filename = "prob_summary.csv"
        df_prob_summary.to_csv(csv_filename)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers based on config
        """
        opt_config = self.hparams.config["training"]["optimizer"]
        scheduler_config = self.hparams.config["training"]["scheduler"]

        # Get optimizer
        optimizer_name = opt_config.get("name", "Adam")
        lr = opt_config.get("lr", 0.001)
        weight_decay = opt_config.get("weight_decay", 0.0)

        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Get scheduler
        scheduler_name = scheduler_config.get("name", "CosineAnnealingLR")
        interval = scheduler_config.get("interval", "step")
        print(f"Total number of steps: {self.trainer.estimated_stepping_batches}")

        if scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.trainer.estimated_stepping_batches)
        elif scheduler_name == "CosineAnnealingLRwithWarmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.trainer.estimated_stepping_batches // 20,
                                                        num_training_steps=self.trainer.estimated_stepping_batches
                                                        )
        elif scheduler_name == "Paper":
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.hparams.config['training']['max_epochs']
            scheduler1 = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=10 * steps_per_epoch
            )
            scheduler2 = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-5
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[10 * steps_per_epoch]
            )
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
            interval = 'epoch'  # ReduceLROnPlateau must use epoch interval
        elif scheduler_name == "StepLR":
            step_size = scheduler_config.get("step_size", 10)
            gamma = scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": interval
            }
        }
    
    def make_rsna_slice_dataloaders(config):
    data_config = config["data"]
    data_root_path = Path(data_config["root_path"])
    trn_csv_path = Path(data_config["train_csv"])
    val_csv_path = Path(data_config["val_csv"])
    num_workers = data_config.get("num_workers", 0)
    num_workers = os.cpu_count() if num_workers == -1 else num_workers

    dataloader_config = config["dataloader"]
    batch_size = dataloader_config.get("batch_size", 32)
    persistent_workers = dataloader_config.get("persistent_workers", True) if num_workers > 0 else False

    trn_df = pd.read_csv(trn_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Identify label columns (all columns except 'filename')
    label_columns = [col for col in trn_df.columns if col != 'filename' and col != 'PatientID']
    print(f"Identified label columns: {label_columns}")

    # Update config with class names if not already set
    if "class_names" not in config:
        config["class_names"] = label_columns

    # Create the list directly, assuming all files exist.
    trn_dataset_list = [
        {
            "image": str(data_root_path / row.filename),
            "label": np.array([getattr(row, col) for col in label_columns], dtype=float)
        }
        for row in trn_df.itertuples(index=False)
    ]

    label_columns_val = [col for col in val_df.columns if col != 'filename' and col != 'PatientID']
    if label_columns != label_columns_val:
        print("Warning: Training and validation label columns differ!")
        raise RuntimeError

    # Create the validation list directly
    val_dataset_list = [
        {
            "image": str(data_root_path / row.filename),
            "label": np.array([getattr(row, col) for col in label_columns], dtype=float)
        }
        for row in val_df.itertuples(index=False)
    ]

    # --- Sanity Check ---
    if len(trn_dataset_list) != len(trn_df):
        print(f"Warning: Training list length ({len(trn_dataset_list)}) doesn't match CSV rows ({len(trn_df)}).")
    if len(val_dataset_list) != len(val_df):
        print(f"Warning: Validation list length ({len(val_dataset_list)}) doesn't match CSV rows ({len(val_df)}).")

    print(f"Constructed train dataset list with {len(trn_dataset_list)} items (expected {len(trn_df)}).")
    print(f"Constructed val dataset list with {len(val_dataset_list)} items (expected {len(val_df)}).")

    train_transforms = get_rsna_classification_transforms(config, is_train=True)
    val_transforms = get_rsna_classification_transforms(config, is_train=False)

    trn_dataset = PersistentDataset(
        data=trn_dataset_list,
        transform=train_transforms,
        cache_dir=data_config.get("cache_dir_train", f"cache_dir/trn"),
    )
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )

    val_dataset = PersistentDataset(
        data=val_dataset_list,
        transform=val_transforms,
        cache_dir=data_config.get("cache_dir_val", "cache_dir/val")
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )

    print(f"Train dataloader ready with {len(trn_dataset)} samples.")
    print(f"Val dataloader ready with {len(val_dataset)} samples.")

    return trn_dataloader, val_dataloader