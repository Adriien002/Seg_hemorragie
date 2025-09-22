from monai.data import DataLoader, PersistentDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import data.dataset as dataset
import data.transform as T_seg

import config

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


def save_prediction_as_nifti(prediction, filename, save_dir, affine=None):
    """Sauvegarde une prédiction de segmentation au format NIfTI"""
    # Appliquer softmax pour obtenir les probabilités
    pred_probs = torch.softmax(prediction, dim=1)
    
    # Convertir en masque de segmentation (argmax pour obtenir la classe avec la plus haute probabilité)
    pred_mask = torch.argmax(pred_probs, dim=1).squeeze().numpy()
    
    # Créer l'affine par défaut si non fourni
    if affine is None:
        affine = np.eye(4)
    elif torch.is_tensor(affine):
        affine = affine.squeeze().numpy()
    
    # Créer l'image NIfTI avec le masque de segmentation
    nifti_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine)
    
    # Nom du fichier de sortie
    base_name = filename.replace('.nii.gz', '').replace('.nii', '')
    output_path = os.path.join(save_dir, f"{base_name}_segmentation.nii.gz")
    
    # Sauvegarder
    nib.save(nifti_img, output_path)
    print(f"Masque de segmentation sauvegardé : {output_path}")
    
    return output_path
# Load data (same as original)


predict_files = dataset.get_data_files(f"{config['dataset']['dataset_dir']}/val/img",
                                       f"{config['dataset']['dataset_dir']}/val/seg") #same as val
    
predict_dataset = PersistentDataset(
        predict_files,
        transform=T_seg.get_val_transforms(config),
        cache_dir=os.path.join(config['dataset']['save_dir'], "cache_test")  
    )
    
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False,  num_workers=4  )




#Configure trainer (no need for checkpoint callback and logger here)

trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        #check_val_every_n_epoch=5,
        accelerator="auto",
        devices=[0],
        default_root_dir=config['dataset']['save_dir'],
        #callbacks=callbacks,
        
    )

print(f"Chargement du modèle depuis : {config}")
model = HemorrhageModel.load_from_checkpoint(
        config['CHECKPOINT_PATH'],
        num_steps=1  # Pas important pour l'inférence
    )
    # Start training
 # Lancement de l'inférence
print("Inference begining")
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
        
        # Sauvegarder le masque de segmentation au format NIfTI
        save_prediction_as_nifti(
            batch_result['preds'], 
            batch_result['filename'], 
            os.path.join(config['dataset']['save_dir'], "predictions"),
            batch_result.get('affine', None)
        )
        
        # for class_name, score in batch_result['dice'].items():
        #     print(f"{class_name}: {score:.4f}")
        #     if not np.isnan(score):
        #         all_dice_scores[class_name].append(score)
    
#     # Calcul des moyennes
# print("\n" + "="*50)
# print("Mean per classe")
# print("="*50)
    
# overall_mean = []
# for class_name, scores in all_dice_scores.items():
#         valid_scores = [score for score in scores if not np.isnan(score)]
#         if valid_scores:  # If score not nan
#             mean_score = np.mean(scores)
#             std_score = np.std(scores)
#             print(f"{class_name}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})")
#             overall_mean.append(mean_score)
#         else:
#             print(f"{class_name}: No prediction")
    
# if overall_mean:
#         print(f"\n Mean {np.mean(overall_mean):.4f}")
    
# print(f"\nSaving results in  : {config.SAVE_DIR}")

