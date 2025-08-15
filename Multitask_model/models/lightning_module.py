from models.architecture import BasicUNetWithClassification
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelRecall, MultilabelAUROC, MultilabelPrecision
from transformers import get_linear_schedule_with_warmup
import torch
import config
import torch.nn as nn



class MultiTaskHemorrhageModule(pl.LightningModule):
    def __init__(self, num_steps: int, seg_weight: float = 1.0, cls_weight: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_steps = num_steps
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
   
        
        # Modèle multi-tâche
        self.model = BasicUNetWithClassification(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,  # pour segmentation
            num_cls_classes=config.NUM_CLASSES  # pour classification
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
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
   
    
    def _get_class_lossfn(self):
        pos_weights = torch.tensor([1.0] * config.NUM_CLASSES, dtype=torch.float)
        pos_weights = pos_weights.to(self.device)  

        print(f"Répartition des poids : {pos_weights}")
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    
      
    def forward(self, x):
        return self.model(x)
        
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
            
            total_loss+=loss_cls
           

    #     # Log etc.

        if batch["segmentation"] is not None:
            x_seg = batch["segmentation"]["image"]
            y_seg = batch["segmentation"]["label"]

        # Forward pass
          
            seg_logits,_ = self.model(x_seg)

        # Loss segmentation
            loss_seg = self.seg_loss_fn(seg_logits, y_seg)
            total_loss += loss_seg
       

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
            self.log("train_cls_loss", loss_cls, on_step=True, on_epoch=True)
      

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
                
                total_loss += loss_cls
                
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
                
                total_loss += loss_seg

                
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

            for i in range(config.NUM_CLASSES):
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



class MultiTaskHemorrhageModule_gradnorm(pl.LightningModule):
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
            num_cls_classes=config.NUM_CLASSES  # pour classification
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
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
        
        # Optimizers
        
        self.optimiser_model = None  # Initialisé dans configure_optimizers
        self.optimiser_w = None  # Initialisé dans configure_optimizers
   
    
    def _get_class_lossfn(self):
        pos_weights = torch.tensor([1.0] * config.NUM_CLASSES, dtype=torch.float)
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

            for i in range(config.NUM_CLASSES):
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
    
    
    
class MultiTaskHemorrhageModule_homeo(pl.LightningModule):
    def __init__(self, num_steps: int, log_sigma_seg: float = 0.0, log_sigma_cls: float = 0.0 , pos_weights= None):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_steps = num_steps
        self.log_sigma_seg = nn.Parameter(torch.tensor(0.0))  # log(σ_seg^2)
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))  # log(σ_cls^2)
        
        # Modèle multi-tâche
        self.model = BasicUNetWithClassification(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,  # pour segmentation
            num_cls_classes=config.NUM_CLASSES  # pour classification
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
        self.cls_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=config.NUM_CLASSES)
        self.cls_mean_precision = MultilabelPrecision(num_labels=config.NUM_CLASSES, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=config.NUM_CLASSES, threshold=0.5)
        
    def compute_uncertainty_loss(self, loss_seg, loss_cls):
        loss_seg_weighted = torch.exp(-self.log_sigma_seg) * loss_seg + 0.5 * self.log_sigma_seg
        loss_cls_weighted = torch.exp(-self.log_sigma_cls) * loss_cls + self.log_sigma_cls
        
        return loss_seg_weighted + loss_cls_weighted
    
    def _get_class_lossfn(self):
        pos_weights = torch.tensor([1.0] * config.NUM_CLASSES, dtype=torch.float)
        pos_weights = pos_weights.to(self.device)  

        print(f"Répartition des poids : {pos_weights}")
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    
      
    def forward(self, x):
        return self.model(x)
        
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
           

    #     # Log etc.

        if batch["segmentation"] is not None:
            x_seg = batch["segmentation"]["image"]
            y_seg = batch["segmentation"]["label"]

        # Forward pass
          
            seg_logits,_ = self.model(x_seg)

        # Loss segmentation
            loss_seg = self.seg_loss_fn(seg_logits, y_seg)
            
        if loss_seg is not None and loss_cls is not None:
            total_loss = self.compute_uncertainty_loss(loss_seg, loss_cls)
        elif loss_seg is not None:
            total_loss = torch.exp(-self.log_sigma_seg) * loss_seg + 0.5 * self.log_sigma_seg
        elif loss_cls is not None:
            total_loss = torch.exp(-self.log_sigma_cls) * loss_cls + self.log_sigma_cls
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
            self.log("train_cls_loss", loss_cls, on_step=True, on_epoch=True)
          
        # Suivi des pindérations  
        self.log("log_sigma_seg", self.log_sigma_seg, prog_bar=True)
        self.log("log_sigma_cls", self.log_sigma_cls, prog_bar=True)

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

                
                # Log total loss
        batch_size = 0
        if batch["classification"] is not None:
                batch_size += batch["classification"]["image"].shape[0]
        if batch["segmentation"] is not None:
                batch_size += batch["segmentation"]["image"].shape[0]
                
                
        if loss_seg is not None and loss_cls is not None:
            total_loss = torch.exp(-self.log_sigma_seg) * loss_seg + 0.5 * self.log_sigma_seg + torch.exp(-self.log_sigma_cls) * loss_cls + self.log_sigma_cls
        elif loss_seg is not None:
            total_loss = torch.exp(-self.log_sigma_seg) * loss_seg + 0.5 * self.log_sigma_seg
        elif loss_cls is not None:
            total_loss = torch.exp(-self.log_sigma_cls) * loss_cls + self.log_sigma_cls

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

            for i in range(config.NUM_CLASSES):
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