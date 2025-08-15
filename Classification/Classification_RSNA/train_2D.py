import pandas as pd
import numpy as np
from pathlib import Path
import monai.transforms as T
import torch
import torch.nn as nn
from torchvision.models import resnet18
from monai.networks.nets import densenet121, SEResNet50, ResNet
from monai.transforms import Compose, Resize, ToTensor
from monai.data import DataLoader, PersistentDataset, Dataset
from tqdm import tqdm
import os
from timeit import default_timer as timer
import gc
from torchmetrics.classification import MultilabelAUROC, MultilabelSpecificity, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

# Configuration
NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def print_train_time(start: float, end: float, device: torch.device = None):
    """Print training time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def compute_accuracy(y_pred, y_true, threshold=0.5):
    """
    Renvoie l'accuracy multilabel (exact match pour chaque label indépendamment).
    """
    preds = torch.sigmoid(y_pred) > threshold
    correct = (preds == y_true.bool()).float()
    return correct.mean().item()


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim,
               compute_accuracy,
               device: torch.device):
    """Training step for one epoch"""
    train_loss, train_acc = 0, 0
    model.train()

    for i, batch in enumerate(dataloader):
        X = batch["image"].to(device)
        y = batch["label"].to(device)
    
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += compute_accuracy(y_pred=y_pred, y_true=y)

        # 3. Optimizer zero grad 
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
      
        if (i % 400 == 0):
            print(f"Looked at {i * len(X)}/{len(dataloader.dataset)} samples")
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.4f}")
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             compute_accuracy,
             device: torch.device):
    """Validation step"""
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            X_test = batch["image"].to(device)
            y_test = batch["label"].to(device)

            # 1. Forward pass
            test_pred = model(X_test)

            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y_test)
            test_loss += loss
            test_acc += compute_accuracy(y_true=y_test, y_pred=test_pred)
     
            if (batch_idx % 400 == 0):
                print(f"Looked at {batch_idx * len(X_test)}/{len(dataloader.dataset)} samples")

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.4f}")
    
    return test_loss, test_acc




def create_model(num_classes, device):
    """Create and return the model"""
    model = ResNet(
        block='basic',           # BasicBlock for ResNet18/34
        layers=[2, 2, 2, 2],    # ResNet18 architecture
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=2,
        n_input_channels=1,     # Input = Scan
        num_classes=num_classes,
        conv1_t_size=7,
        conv1_t_stride=2
    )
    model.to(device)
    return model


def create_transforms():
    """Create training transforms"""
    window_preset = {"window_center": 40, "window_width": 80}
    
    train_transforms = T.Compose([
        T.LoadImaged(keys=["image"], image_only=True),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
            a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        T.EnsureChannelFirstd(keys=["image"]),
        T.ResizeD(keys=["image"], spatial_size=(224, 224)),
        T.ToTensord(keys=["image", "label"])  
    ])
    
    return train_transforms


# def prepare_data(csv_path, dicom_dir, label_cols):
#     """Prepare data list for MONAI"""
#     df = pd.read_csv(csv_path)
    
#     data_list = [
#         {
#             "image": str(dicom_dir / row['filename']),
#             "label": np.array([row[col] for col in label_cols], dtype=np.float32)
#         }
#         for _, row in df.iterrows()
#     ]
    
#     return data_list

def prepare_data(csv_path, dicom_dir, label_cols):
    """Prepare data list for MONAI"""
    df = pd.read_csv(csv_path)
    
   
    data_list = [
        {
            "image": str(dicom_dir / row.filename),
            "label": np.array([getattr(row, col) for col in label_cols], dtype=np.float32)
        }
        for row in df.itertuples(index=False) # itertuples est plus rapide que iterrows?
    ]
    
    return data_list




def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                device: torch.device = DEVICE):
    

    
  
    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        X = batch["image"].to(device)
        y = batch["label"].to(device)
      


        # Forward pass
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss
        
        # zero grad
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        
        
        
        if (i % 400 == 0):
            print(f"Looked at {i * len(X)}/{len(dataloader.dataset)} samples")

    
    # Calcul des moyennes finales
    train_loss /= len(dataloader)
    return train_loss

def val_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = DEVICE):
    """
    CORRECTION 7: Métriques seulement en validation, comme le doctorant
    """
    # Métriques - créées localement pour éviter les fuites
    mean_auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average="macro").to(device)
    mean_recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    mean_precision_metric = MultilabelPrecision(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average=None).to(device)
    
    model.eval()
    val_loss = 0.0
    
    with torch.inference_mode():  # CORRECTION 8: inference_mode au lieu de no_grad
        for i, batch in enumerate(dataloader):
            X = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            
            # Forward pass
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            val_loss += loss
            
            # Conversion en probabilités
            y_probs = torch.sigmoid(y_logits)
            
            # Mise à jour des métriques
            mean_auc_metric.update(y_probs, y.int())
            mean_recall_metric.update(y_probs, y.int())
            mean_precision_metric.update(y_probs, y.int())
            auc_metric.update(y_probs, y.int())
            
            if (i % 400 == 0):
                print(f"Looked at {i * len(X)}/{len(dataloader.dataset)} samples")
    
    # Calcul des moyennes finales
    val_loss /= len(dataloader)
    
    # Compute
    mean_auc = mean_auc_metric.compute()
    mean_recall = mean_recall_metric.compute()
    mean_precision = mean_precision_metric.compute()
    auc_per_class = auc_metric.compute()
    
    # CORRECTION 11: Conversion immédiate en float
    mean_auc_val = mean_auc.item()
    mean_recall_val = mean_recall.item()
    mean_precision_val = mean_precision.item()
    auc_per_class_vals = [auc_per_class[i].item() for i in range(NUM_CLASSES)]
    
    # CORRECTION 12: Reset et suppression explicite des métriques
    mean_auc_metric.reset()
    mean_recall_metric.reset()
    mean_precision_metric.reset()
    auc_metric.reset()
    
    
    # Affichage des résultats
    print(f"\nValidation Loss: {val_loss:.5f}")
    print(f"Val Mean AUC: {mean_auc_val:.4f} | Val Mean Recall: {mean_recall_val:.4f} | Val Mean Precision: {mean_precision_val:.4f}")
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: AUC={auc_per_class_vals[i]:.4f}")
    
    return val_loss, mean_auc_val, mean_recall_val, mean_precision_val
    
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):

    history = {
        'train_loss': [],
        'val_loss': [], 
        'val_auc': [], 
        'val_recall': [], 
        'val_precision': []
    }
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize AUC
        factor=0.1,
        patience=3,
        verbose=False      
    )
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        
        # Validation phase
        val_loss, val_auc, val_recall, val_precision = val_epoch(model, val_loader, loss_fn, DEVICE)
        
        scheduler.step(val_auc)
        
        # Sauvegarde de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        
        
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()  # Garbage collection Python
    
    return history
# def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
#     # Historique simplifié - on garde les métriques les plus importantes
#     history = {
#         'train_loss': [],
#         'val_loss': [], 'val_auc': [], 'val_recall': [], 'val_precision': []
#     }
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='max',          # Maximize AUC
#         factor=0.1,          # Dividing 10 times
#         patience=3,
#         verbose=True        
#     )
    
#     for epoch in tqdm(range(epochs), desc="Training Epochs"):
#         print(f"\nEpoch {epoch+1}/{epochs}")
#         print("-" * 50)
        
#         # Training phase
#         train_loss = train_epoch(  
#             model, train_loader, loss_fn, optimizer, scheduler, DEVICE
#         )
        
#         # Validation phase
#         val_loss, val_auc, val_recall, val_precision = val_epoch(  
#             model, val_loader, loss_fn, DEVICE
#         )
        
#         scheduler.step(val_auc)  

#         # Sauvegarde de l'historique
#         history['train_loss'].append(train_loss)
#         # history['train_auc'].append(train_auc)
#         # history['train_recall'].append(train_recall)
#         # history['train_precision'].append(train_precision)
        
#         history['val_loss'].append(val_loss)
#         history['val_auc'].append(val_auc)
#         history['val_recall'].append(val_recall)
#         history['val_precision'].append(val_precision)
        
#         # Nettoyage mémoire GPU périodique
#         if epoch % 10 == 0:
#             torch.cuda.empty_cache()
    
#     return history
  
 
       
    
def main():
    """Main training function"""
    # === Hyperparams ===
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-3
   

    
    print(f"Using device: {DEVICE}")
    
    # === Config ===
    csv_train_path = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/csv/train_fold0.csv")
    csv_val_path = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/csv/val_fold0.csv")
    dicom_dir = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification/Classification_RSNA/data/rsna-intracranial-hemorrhage-detection/stage_2_train")
    train_cache_dir = Path("./persistent_cache2/fold0_train")  
    val_cache_dir = Path("./persistent_cache2/fold0_val")
    
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # === Prepare Data ===
    print("Preparing data...")
    data_train_list = prepare_data(csv_train_path, dicom_dir, label_cols)
    data_val_list = prepare_data(csv_val_path, dicom_dir, label_cols)

    
    # === Create Transforms ===
    train_transforms = create_transforms()
    
    # === Create Dataset ===
    train_dataset = PersistentDataset(
        data=data_train_list,
        transform=train_transforms,
        cache_dir=str(train_cache_dir),
    )

    val_dataset = PersistentDataset(
        data=data_val_list,
        transform=train_transforms,
        cache_dir=str(val_cache_dir),
    )
    # train_dataset = Dataset(data=data_train_list, transform=train_transforms)
    # val_dataset = Dataset(data=data_val_list, transform=train_transforms)


    print(f"training dataset ready with {len(train_dataset)} samples and cached transforms at {train_cache_dir}")
    print(f"validation dataset ready with {len(val_dataset)} samples and cached transforms at {val_cache_dir}")
    
    # === Create DataLoader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

#     train_loader = DataLoader(
#     train_dataset, 
#     batch_size=BATCH_SIZE, 
#     shuffle=True, 
#     num_workers=4,     # ← essentiel
#     pin_memory=False,  # ← pour l’instant, on désactive tout
# )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"Number of Batches in the training dataset: {len(train_loader)}")
    print(f"Number of Batches in the validation dataset: {len(val_loader)}")


    
    # === Create Model ===
    print("Creating model...")
    model = create_model(NUM_CLASSES, DEVICE)
    
    # === Loss Function ===
    pos_weights = torch.tensor([1.0] * NUM_CLASSES, dtype=torch.float).to(DEVICE)
    print(f"Répartition des poids : {pos_weights}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    # for i, batch in enumerate(train_loader):
    #     print(f"Batch {i} loaded")
    
    #     if i * BATCH_SIZE >= 202400:
    #         print(f"✅ At batch {i}, we reached ~sample 102400 without crash")
    #         break

    # # === Training Loop ===
    print("Starting training...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    start_time = timer()
    
    history = train_model(model, train_loader, val_loader, loss_fn, optimizer, EPOCHS)
    end_time = timer()

    print_train_time(start_time, end_time, DEVICE)

    # === Save History ===
    history_df = pd.DataFrame(history)
    history_save_path = Path("./history/resnet18_fold0_history.csv")
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to {history_save_path}")

    # # === Save Model ===
    # model_save_path = Path("./models/resnet18_fold0.pth")
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")

       
    
   

if __name__ == "__main__":
    main()