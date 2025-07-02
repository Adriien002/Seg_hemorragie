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

from torchmetrics.classification import MultilabelAUROC, MultilabelSpecificity, MultilabelRecall
import torch

# Configuration
NUM_CLASSES = 6
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device = DEVICE):
    
    # Initialisation des métriques
    auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average=None).to(device)
    specificity_metric = MultilabelSpecificity(num_labels=NUM_CLASSES, threshold=0.5, average=None).to(device)
    recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5, average=None).to(device)
    
    mean_auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average="macro").to(device)
    mean_specificity_metric = MultilabelSpecificity(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    mean_recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    
    model.train()
    train_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        
        # zero grad
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        
        
        y_probs = torch.sigmoid(y_logits)
        
        # Update metrics
        auc_metric.update(y_probs, y.int())
        specificity_metric.update(y_probs, y.int())
        recall_metric.update(y_probs, y.int())
        
        mean_auc_metric.update(y_probs, y.int())
        mean_specificity_metric.update(y_probs, y.int())
        mean_recall_metric.update(y_probs, y.int())
        
        if batch % 400 == 0:
            print(f"Traité {batch * len(X)}/{len(dataloader.dataset)} échantillons")
    
    # Calcul des moyennes finales
    train_loss /= len(dataloader)
    auc_per_class = auc_metric.compute()
    specificity_per_class = specificity_metric.compute()
    recall_per_class = recall_metric.compute()
    
    mean_auc = mean_auc_metric.compute()
    mean_specificity = mean_specificity_metric.compute()
    mean_recall = mean_recall_metric.compute()
    
    # Affichage des résultats
    print(f"\nTrain Loss: {train_loss:.5f}")
    print(f"Mean AUC: {mean_auc:.4f} | Mean Specificity: {mean_specificity:.4f} | Mean Recall: {mean_recall:.4f}")
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: AUC={auc_per_class[i]:.4f}, Spec={specificity_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}")
    
    return train_loss, mean_auc.item(), mean_specificity.item(), mean_recall.item()

def val_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = DEVICE):
    
    # Initialisation des métriques (même structure que pour l'entraînement)
    auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average=None).to(device)
    specificity_metric = MultilabelSpecificity(num_labels=NUM_CLASSES, threshold=0.5, average=None).to(device)
    recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5, average=None).to(device)
    
    mean_auc_metric = MultilabelAUROC(num_labels=NUM_CLASSES, average="macro").to(device)
    mean_specificity_metric = MultilabelSpecificity(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    mean_recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=0.5, average="macro").to(device)
    
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()
            
            # Conversion en probabilités
            y_probs = torch.sigmoid(y_logits)
            
            # Mise à jour des métriques
            auc_metric.update(y_probs, y.int())
            specificity_metric.update(y_probs, y.int())
            recall_metric.update(y_probs, y.int())
            
            mean_auc_metric.update(y_probs, y.int())
            mean_specificity_metric.update(y_probs, y.int())
            mean_recall_metric.update(y_probs, y.int())
    
    # Calcul des moyennes finales
    val_loss /= len(dataloader)
    auc_per_class = auc_metric.compute()
    specificity_per_class = specificity_metric.compute()
    recall_per_class = recall_metric.compute()
    
    mean_auc = mean_auc_metric.compute()
    mean_specificity = mean_specificity_metric.compute()
    mean_recall = mean_recall_metric.compute()
    
    # Affichage des résultats
    print(f"\nValidation Loss: {val_loss:.5f}")
    print(f"Val Mean AUC: {mean_auc:.4f} | Val Mean Spec: {mean_specificity:.4f} | Val Mean Recall: {mean_recall:.4f}")
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: AUC={auc_per_class[i]:.4f}, Spec={specificity_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}")
    
    return val_loss, mean_auc.item(), mean_specificity.item(), mean_recall.item()


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    history = {
        'train_loss': [], 'train_auc': [], 'train_spe': [], 'train_rec': [],
        'val_loss': [], 'val_auc': [], 'val_spe': [], 'val_rec': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss, train_auc, train_spe, train_rec = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE
        )
        
        # Validation phase
        # print("Validation phase...")
        # val_loss, val_auc, val_spe, val_rec = val_epoch(
        #     model, val_loader, loss_fn, DEVICE
        # )
        
        # Sauvegarde de l'historique
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_spe'].append(train_spe)
        history['train_rec'].append(train_rec)
        # history['val_loss'].append(val_loss)
        # history['val_auc'].append(val_auc)
        # history['val_spe'].append(val_spe)
        # history['val_rec'].append(val_rec)
    
    return history


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
        n_input_channels=1,     # Grayscale input
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


def prepare_data(csv_path, dicom_dir, label_cols):
    """Prepare data list for MONAI"""
    df = pd.read_csv(csv_path)
    
    data_list = [
        {
            "image": str(dicom_dir / row['filename']),
            "label": np.array([row[col] for col in label_cols], dtype=np.float32)
        }
        for _, row in df.iterrows()
    ]
    
    return data_list


def main():
    """Main training function"""
    # === Hyperparams ===
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # === Config ===
    csv_path = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification_RSNA/data/csv/train_fold0.csv")
    dicom_dir = Path("/home/tibia/Projet_Hemorragie/Seg_hemorragie/Classification_RSNA/data/rsna-intracranial-hemorrhage-detection/stage_2_train")
    cache_dir = Path("./persistent_cache/fold0")
    
    label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # === Prepare Data ===
    print("Preparing data...")
    data_list = prepare_data(csv_path, dicom_dir, label_cols)
    
    # === Create Transforms ===
    train_transforms = create_transforms()
    
    # === Create Dataset ===
    train_dataset = PersistentDataset(
        data=data_list,
        transform=train_transforms,
        cache_dir=str(cache_dir),
    )
    
    print(f"Dataset ready with {len(train_dataset)} samples and cached transforms at {cache_dir}")
    
    # === Create DataLoader ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"Number of Batches in the dataset: {len(train_loader)}")
    
    # === Create Model ===
    print("Creating model...")
    model = create_model(NUM_CLASSES, DEVICE)
    
    # === Loss Function ===
    pos_weights = torch.tensor([1.0] * NUM_CLASSES, dtype=torch.float).to(DEVICE)
    print(f"Répartition des poids : {pos_weights}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # === Training Loop ===
    print("Starting training...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    start_time = timer()
    
    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
        print(f"\nEpoch: {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_accuracy=compute_accuracy,
            device=DEVICE
        )
    
    end_time = timer()
    total_train_time = print_train_time(start_time, end_time, DEVICE)
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Total training time: {total_train_time:.3f} seconds")
    print(f"Average time per epoch: {total_train_time/EPOCHS:.3f} seconds")


if __name__ == "__main__":
    main()