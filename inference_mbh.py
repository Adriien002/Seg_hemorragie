import os
import warnings

import pytorch_lightning as pl
import torch
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, DiceHelper
from monai.networks.nets import UNet,SwinUNETR
import monai.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer




# Configuration
DATASET_DIR = '/home/tibia/Projet_Hemorragie/mbh_seg/nii'
CHECKPOINT_PATH = "/home/tibia/Projet_Hemorragie/MBH_swin_log/lightning_logs/version_3/checkpoints/epoch=894-step=69810.ckpt"
SAVE_DIR = "/home/tibia/Projet_Hemorragie/inference"
os.makedirs(SAVE_DIR, exist_ok=True)

test_transforms = T.Compose([
    # Loading transforms
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'), # who cares about the background ?
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),  # make sure all images are the same orientation
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']), # to isotropic spacing
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 32)),  # make sure we have at least 96 slices
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),  # clip images
])


def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]




##DEF DU MODELE
class HemorrhageModel(pl.LightningModule):
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.model = SwinUNETR(
        img_size=(96, 96, 96),  
        in_channels=1,
        out_channels=6,
        feature_size=48,  
        use_checkpoint=True  # Pour économiser la mémoire
)
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True) # don't need to weight the dice ce loss
        self.dice_metric = DiceHelper(include_background=False,
                                      softmax=True,
                                      num_classes=6,
                                      reduction='none')

 
   
    def predict_step(self, batch, batch_idx):
        x, y = batch["image"], batch["seg"]
        
        # Inférence avec sliding window
        y_hat = sliding_window_inference(x,
                                        roi_size=(96, 96, 96),
                                        sw_batch_size=2,
                                        predictor=self.model)
        
      
      
        scores, _ = self.dice_metric(y_hat, y)
        
        dice_scores = {}
        for class_idx in range(5):  # 0-4 correspondant aux classes 1-5
            dice_scores[f"dice_c{class_idx+1}"] = scores[0, class_idx].item()
    
    # Si besoin de la classe 0 (background)
    # dice_scores["dice_c0"] = ... 
    
        return {
        'preds': y_hat,
        'dice': dice_scores,
        'filenames': batch["image_meta_dict"]["filename_or_obj"]
    }
   
   


def main():

    trainer = Trainer(accelerator="gpu", devices=1)
    model = HemorrhageModel.load_from_checkpoint("/home/tibia/Projet_Hemorragie/MBH_swin_log/lightning_logs/version_3/checkpoints/epoch=894-step=69810.ckpt", model=HemorrhageModel)

    predictions = trainer.predict(model, dataloaders=[(input_data, None)])

    test_files = get_data_files(f"{DATASET_DIR}/test/img", f"{DATASET_DIR}/test/seg")

    test_dataset = PersistentDataset(
        test_files,
        transform=test_transforms,
        cache_dir=os.path.join(SAVE_DIR, "cache_train")
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    predictions = trainer.predict(model, dataloaders=(test_loader))


    total_dice = 0.0
    for i, batch_result in enumerate(predictions):
        print(f"Case {i+1} ({batch_result['filenames']}):")
        print(f"Dice score: {batch_result['dice'].item():.4f}")
        total_dice += batch_result['dice'].item()
    
    print(f"\nDice score moyen: {total_dice/len(predictions):.4f}")

if __name__ == "__main__":
    main()







