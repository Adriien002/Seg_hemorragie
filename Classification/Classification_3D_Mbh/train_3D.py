from monai.data import PersistentDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import monai.transforms as T
import torch
import warnings
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelAUROC, MultilabelSpecificity, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

# ---- Config ----
csv_path = Path("/home/tibia/Projet_Hemorragie/MBH_label_case/splits/train_split.csv")
nii_dir = Path("/home/tibia/Projet_Hemorragie/MBH_label_case")
cache_dir = Path("./persistent_cache/3D_train_cache")  
cache_dir.mkdir(parents=True, exist_ok=True)

label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
df = pd.read_csv(csv_path)

# ---- Build MONAI-style data list ----
data_list = [
    {
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32)
    }
    for _, row in df.iterrows()
]

# ---- Transforms CORRIGÉES ----
window_preset = {"window_center": 40, "window_width": 80}

train_transforms = T.Compose([
    # Load image only
    T.LoadImaged(keys=["image"], image_only=True),  
    T.EnsureChannelFirstd(keys=["image"]),
    
    # Harmonisation spatiale
    T.Orientationd(keys=["image"], axcodes='RAS'),
    T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    
   
    T.ResizeWithPadOrCropd(
        keys=["image"], 
        spatial_size=(224, 224, 144),
        mode="constant",  # Padding avec des zéros
        constant_values=0
    ),
    
    # Intensity normalization
    T.ScaleIntensityRanged(
        keys=["image"],
        a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
        a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    # Augmentations
    T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
    T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
    T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

    # Final tensor
    T.ToTensord(keys=["image", "label"])
])

# ---- PersistentDataset ----
train_dataset = PersistentDataset(
    data=data_list,
    transform=train_transforms,
    cache_dir=str(cache_dir),
)

print(f"Dataset ready with {len(train_dataset)} samples and cached transforms at {cache_dir}")

# Test pour vérifier les tailles
print("Vérification des tailles des premières images:")
for i in range(min(3, len(train_dataset))):
    sample = train_dataset[i]
    print(f"Image {i}: {sample['image'].shape}, Label: {sample['label'].shape}")

# ---- Suite du code d'entraînement ----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import ResNet
from tqdm import tqdm

# === Hyperparams ===
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
CLASS_NAMES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

print(f"Using device: {DEVICE}")
print(f"Number of Batches in the dataset: {len(train_loader)}")

val_transforms = T.Compose([
    T.LoadImaged(keys=["image"], image_only=True),  
    T.EnsureChannelFirstd(keys=["image"]),
    T.Orientationd(keys=["image"], axcodes='RAS'),
    T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144), mode="constant", constant_values=0),
    T.ScaleIntensityRanged(
        keys=["image"],
        a_min=window_preset["window_center"] - window_preset["window_width"] // 2,
        a_max=window_preset["window_center"] + window_preset["window_width"] // 2,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    T.ToTensord(keys=["image", "label"])
])

# === Validation dataset ===
val_csv_path = Path("/home/tibia/Projet_Hemorragie/MBH_label_case/splits/val_split.csv")
val_df = pd.read_csv(val_csv_path)

val_data_list = [
    {
        "image": str(nii_dir / f"{row['patientID_studyID']}.nii.gz"),
        "label": np.array([row[col] for col in label_cols], dtype=np.float32)
    }
    for _, row in val_df.iterrows()
]

val_cache_dir = Path("./persistent_cache/3D_val_cache")
val_cache_dir.mkdir(parents=True, exist_ok=True)

val_dataset = PersistentDataset(
    data=val_data_list,
    transform=val_transforms,  # même pipeline que pour le train, sans data aug si tu veux
    cache_dir=str(val_cache_dir),
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

# === Model ===
model = ResNet(
        block='basic',
        layers=[1, 1, 1, 1],        # Beaucoup moins de couches (vs [2,2,2,2])
        block_inplanes=[32, 64, 128, 256],  # Moins de channels (vs [64,128,256,512])
        spatial_dims=3,
        n_input_channels=1,
        num_classes=NUM_CLASSES,
        conv1_t_size=7,
        conv1_t_stride=(2, 2, 2)    # Stride dans les 3 dimensions
    )
model = model.to(DEVICE)

# === Loss ===
pos_weights = torch.tensor([1.0] * NUM_CLASSES, dtype=torch.float).to(DEVICE)
print(f"Répartition des poids : {pos_weights}")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Test du DataLoader
print("Test du DataLoader:")
try:
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print("✅ DataLoader fonctionne correctement!")
except Exception as e:
    print(f" Erreur dans le DataLoader: {e}")

# === Training function (à définir si pas déjà fait) ===
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward 
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = train_loss / len(dataloader)
    print(f'Average training loss: {avg_loss:.4f}')
    return avg_loss

def val_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = DEVICE):
    """
    CORRECTION 7: Métriques seulement en validation
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
    
    # Convert to CPU and numpy for easier handling
    mean_auc_val = mean_auc.item()
    mean_recall_val = mean_recall.item()
    mean_precision_val = mean_precision.item()
    auc_per_class_vals = [auc_per_class[i].item() for i in range(NUM_CLASSES)]
    
    # Reset metrics for next epoch
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


# === Training Loop ===
torch.manual_seed(42)
torch.cuda.manual_seed(42)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS
)

history_df = pd.DataFrame(history)
history_save_path = Path("./history/resnet18_fold0_history.csv")
history_df.to_csv(history_save_path, index=False)
print(f"Training history saved to {history_save_path}")