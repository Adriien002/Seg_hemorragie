import os
import warnings
import numpy as np
import torch

import pytorch_lightning as pl
from monai.data import DataLoader, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from monai.networks.nets import SwinUNETR, UNet
import monai.transforms as T
from pytorch_lightning import Trainer

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
os.environ["PYTHONWARNINGS"] = "ignore"

# Configuration
DATASET_DIR = '/home/tibia/Projet_Hemorragie/Seg_hemorragie/split_MONAI/'
CHECKPOINT_PATH = "/home/tibia/Projet_Hemorragie/v2_log_test/lightning_logs/version_0/checkpoints/epoch=999-step=156000.ckpt"
SAVE_DIR = "/home/tibia/Projet_Hemorragie/inference"
os.makedirs(SAVE_DIR, exist_ok=True)

test_transforms = T.Compose([
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']),
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
])

def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]

class HemorrhageModel(pl.LightningModule):
    def __init__(self, num_steps=1):  # Valeur par défaut pour l'inférence
        super().__init__()
        self.num_steps = num_steps
        self.model = SwinUNETR(
        img_size=(96, 96, 96),  
        in_channels=1,
        out_channels=4,
        feature_size=48,  # Réduire à 24 si mémoire insuffisante
        use_checkpoint=True  # Pour économiser la mémoire
        )
        
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.dice_metric = DiceHelper(
            include_background=False,
            softmax=True,
            num_classes=4,
            reduction='none'
        )

    def predict_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]
        
        # Inférence avec sliding window
        y_hat = sliding_window_inference(
            x,
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=self.model
        )
        
        # Calcul des scores Dice
        scores, _ = self.dice_metric(y_hat, y)
        
        # Extraction des scores par classe
        dice_scores = {}
        for class_idx in range(3):  # Classes 1-3
            dice_scores[f"dice_c{class_idx+1}"] = scores[0, class_idx].item()
        
        if isinstance(batch["image"].meta["filename_or_obj"], list):
            full_path = batch["image"].meta["filename_or_obj"][0]
        else:
            full_path = batch["image"].meta["filename_or_obj"]
        
        filename = os.path.basename(full_path)
        
        return {
            'preds': y_hat.cpu(),  
            'dice': dice_scores,
            'filename': filename,
            'ground_truth': y.cpu()
        }

def main():
    # Configuration du trainer
    trainer = Trainer(
        accelerator="gpu", 
        devices=[0],
        logger=False,  # Désactiver les logs pour l'inférence
        enable_progress_bar=True
    )
    
    # Chargement du modèle depuis le checkpoint
    print(f"Chargement du modèle depuis : {CHECKPOINT_PATH}")
    model = HemorrhageModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        num_steps=1  # Pas important pour l'inférence
    )
    
    # Préparation des données de test
    test_files = get_data_files(f"{DATASET_DIR}/test/img", f"{DATASET_DIR}/test/seg")
    print(f"Nombre de fichiers de test : {len(test_files)}")
    
    test_dataset = PersistentDataset(
        test_files,
        transform=test_transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_test")  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,  # Pas de shuffle pour l'inférence
        num_workers=4  # Réduit pour éviter les problèmes de mémoire
    )
    
    # Lancement de l'inférence
    print("Inference begining")
    predictions = trainer.predict(model, dataloaders=test_loader)
    
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
        
        for class_name, score in batch_result['dice'].items():
            print(f"{class_name}: {score:.4f}")
            if not np.isnan(score):
                all_dice_scores[class_name].append(score)
    
    # Calcul des moyennes
    print("\n" + "="*50)
    print("Mean per classe")
    print("="*50)
    
    overall_mean = []
    for class_name, scores in all_dice_scores.items():
        valid_scores = [score for score in scores if not np.isnan(score)]
        if valid_scores:  # If score not nan
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{class_name}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})")
            overall_mean.append(mean_score)
        else:
            print(f"{class_name}: No prediction")
    
    if overall_mean:
        print(f"\n Mean {np.mean(overall_mean):.4f}")
    
    print(f"\nSaving results in  : {SAVE_DIR}")

if __name__ == "__main__":
    main()