from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import data.dataset as dataset
import data.transform as T_seg

import config 
from monai.transforms import SaveImage
from models.lightning import HemorrhageModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

import nibabel as nib
import numpy as np

config = config.CONFIG

predict_files = dataset.get_data_files(f"{config['dataset']['dataset_dir']}/val/img",
                                       f"{config['dataset']['dataset_dir']}/val/seg") #same as val
    
predict_dataset = PersistentDataset(
        predict_files,
        transform=T_seg.get_val_transforms(config),
        cache_dir=os.path.join(config['dataset']['save_dir'], "cache_test")  
    )
    
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False,  num_workers=4  )

# Configure trainer (no need for checkpoint callback and logger here)
trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator="auto",
        devices=[0],
        default_root_dir=config['dataset']['save_dir'],
    )

print(f"Chargement du modèle depuis : {config}")
model = HemorrhageModel.load_from_checkpoint(
        config['CHECKPOINT_PATH'],
        num_steps=1  # Pas important pour l'inférence
    )

output_dir = os.path.join(config['dataset']['save_dir'], "predictions")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier de prédictions créé : {output_dir}")

# Utiliser SaveImage (pas SaveImaged) - plus de contrôle
save_pred = SaveImage(
    output_dir=output_dir,
    output_postfix="",# pas de suffixe
    output_ext='.nii.gz',  # Sauvegarder en .nii.gz
    resample=True,              # remettre au spacing/shape original pour challenge
    separate_folder=False,
    print_log=True
)

def clean_metadata_for_saveimage(meta_dict):
    """
    Nettoie les métadonnées pour SaveImage en supprimant les dimensions batch
    """
    cleaned_meta = {}
    for key, value in meta_dict.items():
        if torch.is_tensor(value):
            # Pour TOUS les tensors avec 3+ dimensions - les réduire
            if value.ndim >= 3:
                print(f"Correction de {key}: {value.shape} -> {value[0].shape}")
                cleaned_meta[key] = value[0].cpu().numpy()
            elif value.ndim == 2:
                # Pour les matrices 2D, vérifier si c'est [1, N] 
                if value.shape[0] == 1:
                    cleaned_meta[key] = value[0].cpu().numpy()
                else:
                    cleaned_meta[key] = value.cpu().numpy()
            elif value.ndim == 1:
                cleaned_meta[key] = value.cpu().numpy()
            else:  # scalaire
                cleaned_meta[key] = value.item()
        elif isinstance(value, np.ndarray):
            # Même logique pour numpy arrays
            if value.ndim >= 3:
                print(f"Correction numpy de {key}: {value.shape} -> {value[0].shape}")
                cleaned_meta[key] = value[0]
            elif value.ndim == 2 and value.shape[0] == 1:
                cleaned_meta[key] = value[0]
            else:
                cleaned_meta[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            # Pour les listes/tuples avec un seul élément (batch size 1)
            cleaned_meta[key] = value[0]
        else:
            cleaned_meta[key] = value
    
    # S'assurer que original_affine existe
    if 'original_affine' not in cleaned_meta and 'affine' in cleaned_meta:
        cleaned_meta['original_affine'] = cleaned_meta['affine']
    
    return cleaned_meta

# Lancement de l'inférence
print("Inference beginning")
predictions = trainer.predict(model, dataloaders=predict_loader)
    
# Traitement des résultats
all_dice_scores = {f"dice_c{i+1}": [] for i in range(5)}
    
print("\n" + "="*50)
print("Inference results")
print("="*50)
    
for i, batch_result in enumerate(predictions):
    if batch_result is None:
        print(f"Empty results for batch {i}")
        continue
    
    print(f"\nFile: {batch_result['filename']}")
    print("-" * 30)
    
    # Nettoyer les métadonnées COMPLÈTEMENT
    original_meta = batch_result.get("image_meta_dict", {})
    cleaned_meta = clean_metadata_for_saveimage(original_meta)
    
    print(f"Meta keys: {list(cleaned_meta.keys())}")
    print(f"Affine shape: {cleaned_meta['affine'].shape if cleaned_meta.get('affine') is not None else 'None'}")
    print(f"Original shape: {cleaned_meta.get('spatial_shape', 'N/A')}")
    print(f"Original affine shape: {cleaned_meta['original_affine'].shape if cleaned_meta.get('original_affine') is not None else 'None'}")
    
    # Préparer la prédiction
    pred_discrete = torch.argmax(batch_result['preds'], dim=1)
    
    # Enlever la dimension batch si elle existe
    if pred_discrete.dim() == 4 and pred_discrete.shape[0] == 1:
        pred_discrete = pred_discrete[0]  # [1, H, W, D] -> [H, W, D]
    
    print(f"Pred shape after cleanup: {pred_discrete.shape}")
    
    try:
        # Sauvegarde avec SaveImage (pas SaveImaged)
        # SaveImage prend img et meta_data séparément
        save_pred(img=pred_discrete.cpu(), meta_data=cleaned_meta)
        print(f"Fichier sauvegardé : {batch_result['filename']}")
        
        # Afficher les scores Dice
        for class_name, score in batch_result['dice'].items():
            print(f"{class_name}: {score:.4f}")
            all_dice_scores[class_name].append(score)
            
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {batch_result['filename']}: {e}")
        print(f"Debug - Pred shape: {pred_discrete.shape}")
        print(f"Debug - Affine shape: {cleaned_meta.get('affine').shape if cleaned_meta.get('affine') is not None else 'None'}")
        
        # Debug plus détaillé si nécessaire
        print("=== DEBUG METADONNEES ===")
        for k, v in cleaned_meta.items():
            if hasattr(v, 'shape'):
                print(f"{k}: shape = {v.shape}, type = {type(v)}")
                if hasattr(v, 'ndim') and v.ndim >= 3:
                    print(f"  PROBLEME: {k} a encore {v.ndim} dimensions!")

# Calcul des scores moyens
print("\n" + "="*50)
print("SCORES MOYENS PAR CLASSE")
print("="*50)
for class_name, scores in all_dice_scores.items():
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{class_name}: {mean_score:.4f} ± {std_score:.4f}")