# import models.architecture as Arch
# import os
# import pytorch_lightning as pl
# from monai.inferers import sliding_window_inference
# from monai.losses import DiceCELoss
# from monai.metrics import DiceHelper
# from torch.optim import SGD, Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
# from transformers import get_linear_schedule_with_warmup
# import torch
# import config
# import torch.nn as nn



# class MultiTaskHemorrhageModule(pl.LightningModule):
#     def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float =1.0):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.num_steps = num_steps
#         self.seg_weight = seg_weight
#         self.cls_weight = cls_weight
        
   
        
#         # Modèle multi-tâche
#         self.model =Arch.BasicUNetWithClassification(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=6,  # pour segmentation
#             num_cls_classes=config.NUM_CLASSES  # pour classification
#         )
        
#         # Fonctions de perte
#         self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
#         self.cls_loss_fn = self._get_class_lossfn()  # BCEWithLogitsLoss avec poids de classe
        
#         # Métriques de segmentation
#         self.seg_dice_metric= DiceHelper(
#             include_background=False,
#             softmax=True,
#             num_classes=6,
#             reduction='none'
#         )
        
#         # ajout dice pour in_house
#         # self.seg_dice_metric_inhouse = DiceHelper(
#         #     include_background=False,
#         #     softmax=True,
#         #     num_classes=4,
#         #     reduction='none'
#         # )
        
        
        
#         # Métriques de classification
#         self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
#         self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
#         self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
#         self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
   
    
#     def _get_class_lossfn(self):
#         pos_weights = torch.tensor([1.0] * config.NUM_CLASSES, dtype=torch.float)
#         pos_weights = pos_weights.to(self.device)  

#         print(f"Répartition des poids : {pos_weights}")
#         return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    
      
#     def forward(self, x,task="segmentation"):
#         return self.model(x,task=task)
        
#     def training_step(self, batch, batch_idx):
#         total_loss = 0.0
#         loss_cls = None
#         loss_seg = None

#         if batch["classification"] is not None:
#             x_cls = batch["classification"]["image"]
#             y_cls = batch["classification"]["label"]

#         # Forward pass
#             _ , cls_logits = self.model(x_cls, task="classification")

#         # Loss classification
#             loss_cls = self.cls_loss_fn(cls_logits, y_cls)
            
#             total_loss+= self.cls_weight * loss_cls
#            # AJOUTE ÇA POUR DEBUGGER :
#             #print(f"DEBUG CLS: logits shape {cls_logits.shape}, target shape {y_cls.shape}")
    
   
   

#         #batch_seg

#         if batch["segmentation"] is not None:
#             x_seg = batch["segmentation"]["image"]
#             y_seg = batch["segmentation"]["label"]

#         # Forward pass
          
#             seg_logits,_ = self.model(x_seg, task="segmentation")

#         # Loss segmentation mbh
#             loss_seg = self.seg_loss_fn(seg_logits, y_seg)
#             total_loss += loss_seg
            
#        # Loss segmentation in-house 

        
#         batch_size = 0
#         if batch["classification"] is not None:
#             batch_size += batch["classification"]["image"].shape[0]
#         if batch["segmentation"] is not None:
#             batch_size += batch["segmentation"]["image"].shape[0]

#         # Log the overall training loss, which combines segmentation and classification losses.
#         self.log("train_loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
#         # Log the segmentation loss separately for monitoring its contribution to the overall loss.
#         if loss_seg is not None:
#             self.log("train_seg_loss", loss_seg, batch_size=batch_size, on_step=True, on_epoch=True)

#         # Log the classification loss separately for monitoring its contribution to the overall loss.
#         if loss_cls is not None:
#             self.log("train_cls_loss", loss_cls, on_step=True, on_epoch=True)
      

#         return total_loss
    
        
#     def validation_step(self, batch,batch_idx):
        
#         total_loss = 0.0
#         loss_cls = None
#         loss_seg = None
        
#         if batch["classification"] is  not None : 
#                 x_cls = batch["classification"]["image"]
#                 y_cls = batch["classification"]["label"]
                
#                 _, y_hat_cls = self.model(x_cls, task="classification")
#                 loss_cls = self.cls_loss_fn(y_hat_cls, y_cls)
#                 y_cls_pred = torch.sigmoid(y_hat_cls).as_tensor()
#                 self.cls_auc.update(y_cls_pred, y_cls.int())
#                 self.cls_mean_auc.update(y_cls_pred, y_cls.int())
#                 self.cls_mean_precision.update(y_cls_pred, y_cls.int())
#                 self.cls_mean_recall.update(y_cls_pred, y_cls.int())
                
#                 total_loss += self.cls_weight * loss_cls
        
        
               
#         if batch["segmentation"] is not None:
#                 x_seg = batch["segmentation"]["image"]
#                 y_seg = batch["segmentation"]["label"]
                
#                 y_hat_seg = sliding_window_inference(
#                     x_seg,
#                     roi_size=(64, 64, 64), # Doit matcher ta taille d'entrainement
#                     sw_batch_size=2,
#                     predictor=lambda x: self.model(x,task="segmentation")[0]
#                 )
                
#                 loss_seg = self.seg_loss_fn(y_hat_seg, y_seg)
#                 scores, _ = self.seg_dice_metric(y_hat_seg, y_seg)
               
#                 y_labels = y_seg.unique().long().tolist()[1:]
#                 scores = {label: scores[0][label - 1].item() for label in y_labels}

#                 metrics = {f'dice_c{label}': score for label, score in scores.items()}
                

#                 self.log_dict(metrics, on_epoch=True, prog_bar=True)
                
#                 total_loss += loss_seg

                
#                 # Log total loss
#         batch_size = 0
#         if batch["classification"] is not None:
#                 batch_size += batch["classification"]["image"].shape[0]
#         if batch["segmentation"] is not None:
#                 batch_size += batch["segmentation"]["image"].shape[0]
                
      
#         if loss_seg is not None:
#             self.log("val_seg_loss", loss_seg, batch_size=batch_size, on_step=True, on_epoch=True)

#         # Log the classification loss separately for monitoring its contribution to the overall loss.
#         if loss_cls is not None:
#             self.log("val_cls_loss", loss_cls, on_step=True, on_epoch=True)
            
            
#         self.log("val_loss", total_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True )    

    


                
    
#     def on_validation_epoch_end(self):
#     # === CLASSIFICATION ===
#         if len(self.cls_auc.preds) > 0:
#             class_auc = self.cls_auc.compute()
#             mean_auc = self.cls_mean_auc.compute()
#             mean_precision = self.cls_mean_precision.compute()
#             mean_recall = self.cls_mean_recall.compute()

#             self.log_dict({
#                 'val_mean_auc': mean_auc,
#                 'val_mean_precision': mean_precision,
#                 'val_mean_recall': mean_recall
#             }, on_epoch=True, prog_bar=True)

#             for i in range(config.NUM_CLASSES):
#                 self.log(f'val_auc_class_{i}', class_auc[i].item(), on_epoch=True)

#             self.cls_auc.reset()
#             self.cls_mean_auc.reset()
#             self.cls_mean_precision.reset()
#             self.cls_mean_recall.reset()
            
            
            
        
#     # def configure_optimizers(self):
#     #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

#     #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     #     optimizer,
#     #     mode='min',
#     #     factor=0.5,
#     #     patience=5,
#     #     verbose=True
#     # )

#     #     return {
#     #     "optimizer": optimizer,
#     #     "lr_scheduler": {
#     #         "scheduler": scheduler,
#     #         "monitor": "val_loss",
#     #         "interval": "epoch",
#     #         "frequency": 1
#     #     }
#     # }
    
    
    
#     def configure_optimizers(self):

#         optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003)
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=0,
#                                                     num_training_steps=self.num_steps)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "frequency": 1,
#                 "interval": 'step'
#             }
#          }


# import models.architecture as Arch
# import os
# import pytorch_lightning as pl
# from monai.inferers import sliding_window_inference
# from monai.losses import DiceCELoss
# from monai.metrics import DiceHelper
# from torch.optim import SGD
# from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
# from transformers import get_linear_schedule_with_warmup
# import torch
# import config
# import torch.nn as nn
# import data.dataset as dataset

# class MultiTaskHemorrhageModule(pl.LightningModule):
#     def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 0.5):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.num_steps = num_steps
#         self.seg_weight = seg_weight
#         self.cls_weight = cls_weight
        
#         # Modèle multi-tâche
#         self.model = Arch.BasicUNetWithClassification(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=6,  # pour segmentation
#             num_cls_classes=config.NUM_CLASSES  # pour classification
#         )
        
#         # --- CORRECTION DEVICE POUR LES POIDS ---
#         # On enregistre les poids comme un buffer du modèle. 
#         # PyTorch Lightning les mettra automatiquement sur le bon GPU.
#         pos_weights = dataset.compute_pos_weights(split="train") # Calcule les poids à partir des données d'entraînement
#         pos_weights = torch.clip(pos_weights, min=1.0, max=10.0) # On clip les poids pour éviter des valeurs extrêmes
#         self.register_buffer("pos_weights", pos_weights)
        
#         # Fonctions de perte
#         self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
#         # On utilise le buffer qu'on vient de créer
#         self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights) 
        
#         # Métriques de segmentation
#         self.seg_dice_metric = DiceHelper(
#             include_background=False,
#             softmax=True,
#             num_classes=6,
#             reduction='none'
#         )
        
#         # Métriques de classification
#         self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
#         self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
#         self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
#         self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
      
#     def forward(self, x, task="segmentation"):
#         return self.model(x, task=task)
        
#     def training_step(self, batch, batch_idx):
#         total_loss = 0.0
#         loss_cls = None
#         loss_seg = None
#         batch_size = 0

#         # === BATCH CLASSIFICATION ===
#         if batch.get("classification") is not None:
#             x_cls = batch["classification"]["image"]
#             # NOUVEAU : On utilise class_label ! 
#             # Le pseudo-masque est dans batch["classification"]["label"], on l'ignore ici.
#             y_cls = batch["classification"]["class_label"] 
#             batch_size += x_cls.shape[0]

#             # Forward pass
#             _, cls_logits = self.model(x_cls, task="classification")

#             # Loss classification
#             loss_cls = self.cls_loss_fn(cls_logits, y_cls)
#             total_loss += self.cls_weight * loss_cls
   
#         # === BATCH SEGMENTATION ===
#         if batch.get("segmentation") is not None:
#             x_seg = batch["segmentation"]["image"]
#             y_seg = batch["segmentation"]["label"]
#             batch_size += x_seg.shape[0]

#             # Forward pass
#             seg_logits, _ = self.model(x_seg, task="segmentation")

#             # Loss segmentation
#             loss_seg = self.seg_loss_fn(seg_logits, y_seg)
#             total_loss += self.seg_weight * loss_seg
            
#         # Logging
#         self.log("train_loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
#         if loss_seg is not None:
#             self.log("train_seg_loss", loss_seg, batch_size=x_seg.shape[0], on_step=True, on_epoch=True)
#         if loss_cls is not None:
#             self.log("train_cls_loss", loss_cls, batch_size=x_cls.shape[0], on_step=True, on_epoch=True)

#         return total_loss
        
#     def validation_step(self, batch, batch_idx):
#         total_loss = 0.0
#         loss_cls = None
#         loss_seg = None
#         batch_size = 0
        
#         # === BATCH CLASSIFICATION ===
#         if batch.get("classification") is not None: 
#             x_cls = batch["classification"]["image"]
#             # NOUVEAU : class_label
#             y_cls = batch["classification"]["class_label"]
#             batch_size += x_cls.shape[0]
            
#             _, y_hat_cls = self.model(x_cls, task="classification")
#             loss_cls = self.cls_loss_fn(y_hat_cls, y_cls)
            
          
#             y_cls_pred = torch.sigmoid(y_hat_cls)#.as_tensor()
# #                
#             self.cls_auc.update(y_cls_pred, y_cls.int())
#             self.cls_mean_auc.update(y_cls_pred, y_cls.int())
#             self.cls_mean_precision.update(y_cls_pred, y_cls.int())
#             self.cls_mean_recall.update(y_cls_pred, y_cls.int())
            
#             total_loss += self.cls_weight * loss_cls
        
#         # === BATCH SEGMENTATION ===
#         if batch.get("segmentation") is not None:
#             x_seg = batch["segmentation"]["image"]
#             y_seg = batch["segmentation"]["label"]
#             batch_size += x_seg.shape[0]
            
#             y_hat_seg = sliding_window_inference(
#                 x_seg,
#                 roi_size=(64, 64, 64),
#                 sw_batch_size=2,
#                 predictor=lambda x: self.model(x, task="segmentation")[0]
#             )
            
#             loss_seg = self.seg_loss_fn(y_hat_seg, y_seg)
#             scores, _ = self.seg_dice_metric(y_hat_seg, y_seg)
           
#             y_labels = y_seg.unique().long().tolist()[1:]
#             scores = {label: scores[0][label - 1].item() for label in y_labels}
#             metrics = {f'dice_c{label}': score for label, score in scores.items()}
            
#             self.log_dict(metrics, on_epoch=True, prog_bar=True)
#             total_loss += self.seg_weight * loss_seg

#         # Logging
#         if loss_seg is not None:
#             self.log("val_seg_loss", loss_seg, batch_size=x_seg.shape[0], on_step=True, on_epoch=True)
#         if loss_cls is not None:
#             self.log("val_cls_loss", loss_cls, batch_size=x_cls.shape[0], on_step=True, on_epoch=True)
            
#         self.log("val_loss", total_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)    

#     def on_validation_epoch_end(self):
#         if len(self.cls_auc.preds) > 0:
#             class_auc = self.cls_auc.compute()
#             mean_auc = self.cls_mean_auc.compute()
#             mean_precision = self.cls_mean_precision.compute()
#             mean_recall = self.cls_mean_recall.compute()

#             self.log_dict({
#                 'val_mean_auc': mean_auc,
#                 'val_mean_precision': mean_precision,
#                 'val_mean_recall': mean_recall
#             }, on_epoch=True, prog_bar=True)

#             for i in range(config.NUM_CLASSES):
#                 self.log(f'val_auc_class_{i}', class_auc[i].item(), on_epoch=True)

#             self.cls_auc.reset()
#             self.cls_mean_auc.reset()
#             self.cls_mean_precision.reset()
#             self.cls_mean_recall.reset()
            
#     def configure_optimizers(self):
#         optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=0.00003)
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=0,
#             num_training_steps=self.num_steps
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "frequency": 1,
#                 "interval": 'step'
#             }
#          }




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
import config
import torch.nn as nn
import data.dataset as dataset

class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_steps = num_steps
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
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
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights) 
        
        # Métriques de segmentation
        # self.seg_dice_metric = DiceHelper(
        #     include_background=False,
        #     softmax=True,
        #     num_classes=6,
        #     reduction='none'
        # )
        

        self.seg_orig_dice = DiceHelper(include_background=False, softmax=True, num_classes=6, reduction='none')
        self.seg_inhouse_dice = DiceHelper(include_background=False, softmax=True, num_classes=4, reduction='none')
        
       
        
        # Métriques de classification
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
      
    def forward(self, x, task="seg_orig"):
        return self.model(x, task=task)
        
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        
        # 1. Classification
        if batch.get("classification") is not None:
            x, y = batch["classification"]["image"], batch["classification"]["class_label"]
            _, logits = self.model(x, task="classification")
            loss = self.cls_loss_fn(logits, y)
            total_loss += self.cls_weight * loss
            self.log("train_cls_loss", loss, batch_size=x.shape[0], on_epoch=True)

        # 2. Segmentation Orig
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            logits, _ = self.model(x, task="seg_orig")
            loss = self.seg_loss_fn_mbh(logits, y)
            total_loss += loss
            self.log("train_seg_orig_loss", loss, batch_size=x.shape[0], on_epoch=True)

        # 3. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            logits, _ = self.model(x, task="seg_inhouse")
            loss = self.seg_loss_fn_inhouse(logits, y)
            total_loss += loss
            self.log("train_seg_inhouse_loss", loss, batch_size=x.shape[0], on_epoch=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        total_loss = 0.0

        # 1. Classification
        if batch.get("classification") is not None: 
            x, y = batch["classification"]["image"], batch["classification"]["class_label"]
            _, logits = self.model(x, task="classification")
            loss = self.cls_loss_fn(logits, y)
            
            y_pred = torch.sigmoid(logits)
            self.cls_auc.update(y_pred, y.int())
            self.cls_mean_auc.update(y_pred, y.int())
            
            total_loss += self.cls_weight * loss
            self.log("val_cls_loss", loss, batch_size=x.shape[0], on_epoch=True)

        # 2. Segmentation Orig
        if batch.get("seg_orig") is not None:
            x, y = batch["seg_orig"]["image"], batch["seg_orig"]["label"]
            y_hat = sliding_window_inference(x, roi_size=(64, 64, 64), sw_batch_size=2, predictor=lambda img: self.model(img, task="seg_orig")[0])
            
            loss = self.seg_loss_fn_mbh(y_hat, y)
            scores, _ = self.seg_orig_dice(y_hat, y)
            
            y_labels = y.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            metrics = {f'dice_c{label}': score for label, score in scores.items()}
            
            self.log_dict(metrics, on_epoch=True, prog_bar=True)
            total_loss += loss
        
            self.log("val_seg_loss", loss, batch_size=x.shape[0])


        # 3. Segmentation In-House
        if batch.get("seg_inhouse") is not None:
            x, y = batch["seg_inhouse"]["image"], batch["seg_inhouse"]["label"]
            y_hat = sliding_window_inference(x, roi_size=(64, 64, 64), sw_batch_size=2, predictor=lambda img: self.model(img, task="seg_inhouse")[0])
            
            loss = self.seg_loss_fn_inhouse(y_hat, y)
            scores, _ = self.seg_inhouse_dice(y_hat, y)
            
            for label in y.unique().long().tolist()[1:]:
                self.log(f'val_inhouse_dice_c{label}', scores[0][label - 1].item())
                
            total_loss +=  loss
            self.log("val_seg_inhouse_loss", loss, batch_size=x.shape[0])

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