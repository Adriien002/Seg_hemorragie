import models.architecture as Arch
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
        self.model =Arch.BasicUNetWithClassification(
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

        # Log the overall training loss, which combines segmentation and classification losses.
        self.log("train_loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log the segmentation loss separately for monitoring its contribution to the overall loss.
        if loss_seg is not None:
            self.log("train_seg_loss", loss_seg, batch_size=batch_size, on_step=True, on_epoch=True)

        # Log the classification loss separately for monitoring its contribution to the overall loss.
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
    def __init__(self, num_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Pour GradNorm
        self.num_steps = num_steps
        
        
        self.alpha = 1.5  # paramètre d’asymétrie GradNorm
        self.task_num = 2

        # Poids de pertes trainables
        self.w_seg = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.w_cls = nn.Parameter(torch.tensor(1.0, requires_grad=True))


        self.initial_losses = {}
   
        
        # Modèle multi-tâche
        self.model = Arch.BasicUNetWithClassification(
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
        #torch.autograd.set_detect_anomaly(True)
        
        loss_cls = None
        loss_seg = None
        # loss_cls = torch.tensor(0.0, device=self.device)
        # loss_seg = torch.tensor(0.0, device=self.device)
        epsilon = torch.tensor(1e-7, device=self.device) # pour éviter la division par zéro des  ratios
        
        # Forward passes
        if batch["classification"] is not None:
            x_cls = batch["classification"]["image"]
            y_cls = batch["classification"]["label"]
            _ ,cls_logits = self.model(x_cls)
            loss_cls = self.cls_loss_fn(cls_logits, y_cls)
            #total_loss+=self.w_cls*loss_cls
           
        if batch["segmentation"] is not None:
            x_seg = batch["segmentation"]["image"]
            y_seg = batch["segmentation"]["label"]
            seg_logits,_ = self.model(x_seg)
            loss_seg = self.seg_loss_fn(seg_logits, y_seg)
            #total_loss += self.w_seg* loss_seg
            
        # 1. Choisir W = dernière couche partagée de l’encodeur
        #shared_params = list(self.model.down_4.parameters())

            # 2. Calculer les normes de gradient pour chaque tâche
        shared_params = [p for p in self.model.down_4.parameters() if p.requires_grad]
        G_seg_norm = torch.tensor(0.0, device=self.device)
        G_cls_norm = torch.tensor(0.0, device=self.device)

        if loss_seg is not None:
            G_seg = torch.autograd.grad(self.w_seg * loss_seg, shared_params, retain_graph=True, create_graph=True, allow_unused=True)
            G_seg_norm = torch.norm(torch.cat([g.reshape(-1) for g in G_seg if g is not None]), 2)
        
            
        if loss_cls is not None:
            G_cls = torch.autograd.grad(self.w_cls * loss_cls, shared_params, retain_graph=True, create_graph=True, allow_unused=True)
            G_cls_norm = torch.norm(torch.cat([g.reshape(-1) for g in G_cls]), 2)
        

            # 3. Moyenne des normes
        G_avg = (G_seg_norm + G_cls_norm) / self.task_num

        # 4. Taux relatifs r_i(t)
        
        loss_seg_ratio = torch.tensor(1.0, device=self.device)
        loss_cls_ratio = torch.tensor(1.0, device=self.device)

        if (loss_seg is not None) and (self.initial_losses["seg"] > epsilon):
            loss_seg_ratio = loss_seg / (self.initial_losses["seg"] + epsilon)

        if (loss_cls is not None) and (self.initial_losses["cls"] > epsilon):
            loss_cls_ratio = loss_cls / (self.initial_losses["cls"] + epsilon)
        

        # Détacher pour éviter les problèmes de gradient
        loss_seg_ratio = loss_seg_ratio.detach()
        loss_cls_ratio = loss_cls_ratio.detach()
        
        avg_loss_ratio = (loss_seg_ratio + loss_cls_ratio) / 2
        r_seg = loss_seg_ratio / (avg_loss_ratio + epsilon)
        r_cls = loss_cls_ratio / (avg_loss_ratio + epsilon)

            # 5. Cibles de normes
        target_seg = G_avg * (r_seg ** self.alpha)
        target_cls = G_avg * (r_cls ** self.alpha)
    

            # 6. Perte GradNorm
        L_grad = torch.abs(G_seg_norm - target_seg) + torch.abs(G_cls_norm - target_cls)

        
            
            # 7. Mise à jour des poids
        # === Mise à jour des poids (GradNorm) ===
        
        
        
          
        optimizer_model, optimizer_w = self.optimizers()
        # 3. Mise à jour du modèle
       # backward GradNorm loss
        optimizer_w.zero_grad()
        self.manual_backward(L_grad, retain_graph=True)
        optimizer_w.step()

        # backward weighted task loss
        optimizer_model.zero_grad()
        total_loss = 0.0
        if loss_seg is not None:
            total_loss = total_loss + self.w_seg * loss_seg
        if loss_cls is not None:
            total_loss = total_loss + self.w_cls * loss_cls
        self.manual_backward(total_loss)
        optimizer_model.step()
    
        with torch.no_grad():
            T = self.task_num  # Normalisation pour que la somme des poids reste constante
            sum_w = self.w_seg + self.w_cls
            self.w_seg.data = self.w_seg.data / sum_w * T
            self.w_cls.data = self.w_cls.data / sum_w * T
                    
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

        # Log the weights of segmentation and classification losses (w_seg and w_cls)
        # These weights are dynamically adjusted during training to balance the contributions of each task.
        # Monitoring them helps in debugging and understanding the behavior of the GradNorm algorithm.
        self.log("w_seg", self.w_seg, prog_bar=True)
        self.log("w_cls", self.w_cls, prog_bar=True)


        return total_loss
    
    def validation_step(self, batch, batch_idx):
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        loss_seg = None
        loss_cls = None
        batch_size = 0

        # === Classification ===
        if batch.get("classification") is not None:
            x_cls = batch["classification"]["image"].to(device)
            y_cls = batch["classification"]["label"].to(device)
            _, y_hat_cls = self.model(x_cls)
            loss_cls = self.cls_loss_fn(y_hat_cls, y_cls)
            y_cls_pred = torch.sigmoid(y_hat_cls).as_tensor()

            # Mise à jour métriques
            self.cls_auc.update(y_cls_pred, y_cls.int())
            self.cls_mean_auc.update(y_cls_pred, y_cls.int())
            self.cls_mean_precision.update(y_cls_pred, y_cls.int())
            self.cls_mean_recall.update(y_cls_pred, y_cls.int())

            batch_size += x_cls.shape[0]

        # === Segmentation ===
        if batch.get("segmentation") is not None:
            x_seg = batch["segmentation"]["image"].to(device)
            y_seg = batch["segmentation"]["label"].to(device)

            y_hat_seg = sliding_window_inference(
                x_seg,
                roi_size=(96, 96, 96),
                sw_batch_size=2,
                predictor=lambda x: self.model(x)[0]
            )

            loss_seg = self.seg_loss_fn(y_hat_seg, y_seg)

            # Dice métriques
            scores, _ = self.seg_dice_metric(y_hat_seg, y_seg)
            y_labels = y_seg.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            metrics = {f'dice_c{label}': score for label, score in scores.items()}
            self.log_dict(metrics, on_epoch=True, prog_bar=True)

            batch_size += x_seg.shape[0]

        # === Perte totale pondérée (GradNorm) ===
        with torch.no_grad():
            w_seg_val = self.w_seg.detach()
            w_cls_val = self.w_cls.detach()

        if loss_seg is not None:
            total_loss += w_seg_val * loss_seg
        if loss_cls is not None:
            total_loss += w_cls_val * loss_cls

        # Log des pertes individuelles
        if loss_seg is not None:
            self.log("val_seg_loss", loss_seg, batch_size=batch_size, on_step=False, on_epoch=True)
        if loss_cls is not None:
            self.log("val_cls_loss", loss_cls, batch_size=batch_size, on_step=False, on_epoch=True)

        # Log de la perte totale pour EarlyStopping
        self.log("val_loss", total_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss
    
        
    
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
        
        





import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from torchmetrics.classification import MultilabelAUROC, MultilabelPrecision, MultilabelRecall
from monai.inferers import sliding_window_inference
from torch.optim import SGD
from transformers import get_linear_schedule_with_warmup


class MultiTaskSoftSharing(pl.LightningModule):
    def __init__(self, num_steps, seg_weight=1.0, cls_weight=0.5, reg_lambda=0.01, num_classes=6):
        super().__init__()
        self.save_hyperparameters()

        # Deux backbones distincts pour soft parameter sharing
        self.encoder_seg = Arch.BasicUNetEncoder()
        self.encoder_cls = Arch.BasicUNetEncoder()

        self.seg_decoder = Arch.SegDecoder(out_channels=6)
        self.cls_head = Arch.ClsHead(in_channels=256, num_classes=num_classes)

        # Pertes
        self.seg_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cls_loss_fn = self._get_class_lossfn()  # Avec poids si nécessaire

        # Métriques classification
        self.cls_auc = MultilabelAUROC(num_labels=num_classes, average=None)
        self.cls_mean_auc = MultilabelAUROC(num_labels=num_classes)
        self.cls_mean_precision = MultilabelPrecision(num_labels=num_classes, threshold=0.5)
        self.cls_mean_recall = MultilabelRecall(num_labels=num_classes, threshold=0.5)

        # Métriques segmentation
        self.seg_dice_metric = DiceHelper(
            include_background=False,
            softmax=True,
            num_classes=6,
            reduction='none'
        )

        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.reg_lambda = reg_lambda
        self.num_steps = num_steps

    def _get_class_lossfn(self):
        """Optionnel : ajouter des poids de classe comme dans l'original"""
        # Si vous voulez des poids de classe comme dans l'original :
        # pos_weights = torch.tensor([1.0] * self.hparams.num_classes, dtype=torch.float)
        # return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        return nn.BCEWithLogitsLoss()

    def forward(self, x):
        f_seg = self.encoder_seg(x)
        f_cls = self.encoder_cls(x)
        seg_logits = self.seg_decoder(f_seg)
        cls_logits = self.cls_head(f_cls[-1])
        return seg_logits, cls_logits

    def compute_reg_loss(self):
        """Calcule la perte de régularisation pour le soft parameter sharing"""
        reg_loss = 0.0
        for p1, p2 in zip(self.encoder_seg.parameters(), self.encoder_cls.parameters()):
            reg_loss += torch.norm(p1 - p2, p=2)
        return reg_loss

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        loss_seg, loss_cls = None, None
        batch_size = 0

        # Segmentation
        if batch.get("segmentation") is not None:
            x_seg, y_seg = batch["segmentation"]["image"], batch["segmentation"]["label"]
            y_seg = y_seg.as_tensor()

            seg_logits, _ = self.forward(x_seg)
            loss_seg = self.seg_loss_fn(seg_logits, y_seg)
            total_loss += self.seg_weight * loss_seg
            batch_size += x_seg.shape[0]
            self.log("train_seg_loss", loss_seg, batch_size=batch_size, on_step=True, on_epoch=True)

        # Classification
        if batch.get("classification") is not None:
            x_cls, y_cls = batch["classification"]["image"], batch["classification"]["label"]
            _, cls_logits = self.forward(x_cls)
            y_cls = y_cls.as_tensor().float()
            if y_cls.ndim == 2 and y_cls.shape[0] == 1:  # [1,C] -> [C]
                y_cls = y_cls.squeeze(0)
            loss_cls = self.cls_loss_fn(cls_logits, y_cls)
            total_loss += self.cls_weight * loss_cls
            batch_size += x_cls.shape[0]
            self.log("train_cls_loss", loss_cls, batch_size=batch_size, on_step=True, on_epoch=True)

        # Régularisation soft parameter sharing
        reg_loss = self.compute_reg_loss() * self.reg_lambda
        total_loss += reg_loss
        self.log("train_reg_loss", reg_loss, batch_size=batch_size, on_step=True, on_epoch=True)

        self.log("train_loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0
        loss_seg, loss_cls = None, None
        batch_size = 0

        # Classification
        if batch.get("classification") is not None:
            x_cls, y_cls = batch["classification"]["image"], batch["classification"]["label"]
            _, y_hat_cls = self.forward(x_cls)
            y_hat_cls = y_hat_cls.as_tensor()
            y_cls = y_cls.as_tensor()
            
            if y_cls.ndim == 3 and y_cls.shape[1] == 1:      # [B,1,C] -> [B,C]
                y_cls = y_cls.squeeze(1)
            if y_cls.ndim == 2 and y_cls.shape[-1] == 1:     # [B,C,1] -> [B,C] (si jamais)
                y_cls = y_cls.squeeze(-1)
            y_cls = y_cls.float()
            loss_cls = self.cls_loss_fn(y_hat_cls, y_cls)

            y_cls_pred = torch.sigmoid(y_hat_cls)
            self.cls_auc.update(y_cls_pred, y_cls.int())
            self.cls_mean_auc.update(y_cls_pred, y_cls.int())
            self.cls_mean_precision.update(y_cls_pred, y_cls.int())
            self.cls_mean_recall.update(y_cls_pred, y_cls.int())

            total_loss += self.cls_weight * loss_cls
            batch_size += x_cls.shape[0]

        # Segmentation
        if batch.get("segmentation") is not None:
            x_seg, y_seg = batch["segmentation"]["image"], batch["segmentation"]["label"]

            y_hat_seg = sliding_window_inference(
                x_seg,
                roi_size=(96, 96, 96),
                sw_batch_size=2,
                predictor=lambda x: self.forward(x)[0]
            )
            
            y_hat_seg = y_hat_seg.as_tensor()
            y_seg = y_seg.as_tensor()


            loss_seg = self.seg_loss_fn(y_hat_seg, y_seg)
            scores, _ = self.seg_dice_metric(y_hat_seg, y_seg)

            # Log dice par classe
            y_labels = y_seg.unique().long().tolist()[1:]
            scores = {label: scores[0][label - 1].item() for label in y_labels}
            metrics = {f'val_dice_c{label}': score for label, score in scores.items()}
            self.log_dict(metrics, on_epoch=True, prog_bar=True)

            total_loss += self.seg_weight * loss_seg
            batch_size += x_seg.shape[0]

        self.log("val_loss", total_loss, batch_size=batch_size, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_end(self):
        # Vérifier s'il y a eu des données de classification
        if hasattr(self.cls_auc, 'preds') and len(self.cls_auc.preds) > 0:
            class_auc = self.cls_auc.compute()
            mean_auc = self.cls_mean_auc.compute()
            mean_precision = self.cls_mean_precision.compute()
            mean_recall = self.cls_mean_recall.compute()

            self.log_dict({
                'val_mean_auc': mean_auc,
                'val_mean_precision': mean_precision,
                'val_mean_recall': mean_recall
            }, on_epoch=True, prog_bar=True)

            for i, auc in enumerate(class_auc):
                self.log(f'val_auc_class_{i}', auc.item(), on_epoch=True)

            self.cls_auc.reset()
            self.cls_mean_auc.reset()
            self.cls_mean_precision.reset()
            self.cls_mean_recall.reset()

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True, weight_decay=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}