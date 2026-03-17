from monai import transforms as T
import torch
import utils


#Sliding window
#Bien. Padder l'image pour avoir des dimensions multiples de 96
    
    
         
class TaskBasedTransform_V2:
    """
    Applique un pipeline différent selon la tâche : "segmentation" ou "classification".
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedTransform initialized")


        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=True),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'],
                image_key='image',
                label_key='label',
                pos=5.0,
                neg=1.0,
                spatial_size=(64, 64, 64),
                num_samples=2
            ),
            T.RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5)
        ])
        
        self.cls_pipeline = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), #2mm au lieu de 1mm 
            
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,a_max=140, 
                b_min=0.0, b_max=1.0, 
                clip=True),
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
            #T.RandSpatialCropd(keys=["image"], roi_size=(64, 64, 64), random_size=False),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 176), mode="constant"),
            T.ToTensord(keys=["image", "label"])
        ])
    
        # self.random_crop = T.Compose([
        #     T.RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
        #     T.ToTensord(keys=["image", "label"])  
#])

        
    def __call__(self, data):
        print("ouaii onejeiorjeopia")
        task = data["task"]
        if task == "segmentation":
            return self.seg_pipeline(data)
        elif task == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {task}")
        
    # def __call__(self, data):
    #     if data["task"] == "segmentation":
    #         return self.seg_pipeline(data)
        
    #     elif data["task"] == "classification":
    #         processed = self.cls_pipeline(data)
            
    #         patches = []
    #         for _ in range(25):
    #             patch_data = self.random_crop(processed.copy())
                
    #             # Crée un dict complet pour chaque patch
    #             patch_dict = {
    #                 "image": patch_data["image"],
    #                 "label": processed["label"],  # ← Le label est le même pour tous les patches
    #                 "task": "classification"
    #             }
    #             patches.append(patch_dict)  # ← Ajoute le DICT complet
    #         #print(f"CRRREEEEAAAATINNNNGGGG {len(patches)} PATTTTCHHHESSS -> total mem: {print(torch.cuda.memory_reserved() / 1e9) } GB")
    #         return patches
    
    # def __call__(self, data):
    #     if data["task"] == "segmentation":
    #         return self.seg_pipeline(data)
        
    #     elif data["task"] == "classification":
        
    #         processed = self.cls_pipeline(data)
            
    #         patches = utils.extract_sliding_window_patches(processed["image"], processed["label"], overlap=0.25)
    #         #print(f"CRRREEEEAAAATINNNNGGGG {len(patches)} PATTTTCHHHESSS -> total mem: {print(torch.cuda.memory_reserved() / 1e9) } GB")
       
    #         return patches

class TaskBasedValTransform_V2:
    """
    Transformations de validation — une pipeline par tâche, sans augmentation aléatoire.
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedValTransform initialized")
   # Loading transforms
  # Ensure we load both image and segmentation

 # who cares about the background ?
  # make sure all images are the same orientation
   # to isotropic spacing
 # make sure we have at least 96 slices

        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=False),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(64, 64, 64)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            )
        ])
       

        self.cls_pipeline = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,a_max=140, 
                b_min=0.0, b_max=1.0, 
                clip=True),
            #T.RandSpatialCropd(keys=["image"], roi_size=(64, 64, 64), random_size=False),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 176), mode="constant"),
            T.ToTensord(keys=["image", "label"]) ])
   
        
#         self.random_crop = T.Compose([
#             T.RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
#             T.ToTensord(keys=["image", "label"]) 
# ])


    def __call__(self, data):
        print("Pipeline called for task:", data["task"])
        if data["task"] == "segmentation":
            return self.seg_pipeline(data)
        elif data["task"] == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {data['task']}")
        
    # def __call__(self, data):
    #     if data["task"] == "segmentation":
    #         return self.seg_pipeline(data)
        
    #     elif data["task"] == "classification":
    #         processed = self.cls_pipeline(data)
            
    #         patches = []
    #         for _ in range(25):
    #             patch_data = self.random_crop(processed.copy())
                
    #             # Crée un dict complet pour chaque patch
    #             patch_dict = {
    #                 "image": patch_data["image"],
    #                 "label": processed["label"],  # ← Le label est le même pour tous les patches
    #                 "task": "classification"
    #             }
    #             patches.append(patch_dict)  # ← Ajoute le DICT complet
            
    #         return patches
    
    # def __call__(self, data):
    #     if data["task"] == "segmentation":
    #         return self.seg_pipeline(data)
        
    #     elif data["task"] == "classification":
    #         processed = self.cls_pipeline(data)
            
    #         patches = utils.extract_sliding_window_patches(processed["image"], processed["label"], overlap=0.25)
            
    #         return patches
    
    
    
from monai import transforms as T

class TaskBasedTransform_V3:
    def __init__(self, keys=None):
        print(">>> TaskBasedTransform V2 (Guided MTL) initialized")

        # --- PIPELINE DE SEGMENTATION  ---
        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=True),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'], image_key='image', label_key='label',
                pos=5.0, neg=1.0, spatial_size=(64, 64, 64), num_samples=2
            ),
            
            T.RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
            # On s'assure de garder la clé 'class_label' si elle existe
            T.ToTensord(keys=["image", "label"]) 
        ])
        
        # --- NOUVELLE PIPELINE DE CLASSIFICATION (Guidée par pseudo-masque) ---
        self.cls_pipeline = T.Compose([
            # On charge l'image ET le pseudo-masque (clé "label")
            T.LoadImaged(keys=["image", "label"], image_only=True),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            
            # Pixdim à 1mm pour la classification aussi, pour que le crop soit plus précis sur la lésion
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
            
            # MAGIE 2 : utilisation  pseudo-masque pour centrer le patch de 64x64x64 sur la lésion.
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'], image_key='image', label_key='label',
                pos=5.0, neg=1.0, spatial_size=(64, 64, 64), num_samples=2
            ),
            T.DeleteItemsd(keys=["label"]),  # On supprime le pseudo-masque après le crop
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
            
            
            T.ToTensord(keys=["image", "class_label"]) 
        ])
        
        self.cls_pipeline_no_mask = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140,
                                   b_min=0.0, b_max=1.0, clip=True),
            # Crop random — pas de masque guide , on mais que 1 patch par image pour pas exploser la mémoire
            T.RandSpatialCropSamplesd(
                keys=["image"], roi_size=(64, 64, 64),
                num_samples=1, random_size=False
            ),
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
            T.ToTensord(keys=["image", "class_label"])
        ])
        
    def __call__(self, data):
        task = data["task"]
        if task == "segmentation":
            return self.seg_pipeline(data)
        elif task == "classification":
            if data.get("has_mask", True):
                return self.cls_pipeline(data)
            else:
                return self.cls_pipeline_no_mask(data)
        else:
            raise ValueError(f"Tâche inconnue : {task}")

        
        
class TaskBasedValTransform_V3:
    def __init__(self, keys=None):
        print(">>> TaskBasedValTransform V2 initialized")

        self.seg_pipeline = T.Compose([
            T.LoadImaged(keys=["image", "label"], image_only=False),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)), # wtf 
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True)
        ])
       
        self.cls_pipeline = T.Compose([
            # Pareil : on charge le pseudo-masque pour la validation
            T.LoadImaged(keys=["image", "label"], image_only=True),
            T.EnsureChannelFirstd(keys=["image", "label"]),
            T.CropForegroundd(keys=["image", "label"], source_key='image'),
            T.Orientationd(keys=["image", "label"], axcodes='RAS'),
            T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
            
            # On extrait UN SEUL patch ciblé sur le pseudo-masque pour tester la classification
            T.RandCropByPosNegLabeld(
                keys=['image', 'label'], image_key='image', label_key='label',
                pos=1.0, neg=0.0, spatial_size=(64, 64, 64), num_samples=1
            ),
            T.DeleteItemsd(keys=["label"]),  # Supprimer le pseudo-masque
            
            T.ToTensord(keys=["image", "class_label"]) 
        ])
        
        self.cls_pipeline_no_mask = T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140,
                                   b_min=0.0, b_max=1.0, clip=True),
            T.RandSpatialCropSamplesd(keys=["image"], roi_size=(64, 64, 64), num_samples=1),
            T.ToTensord(keys=["image", "class_label"])
        ])

    def __call__(self, data):
        task = data["task"]
        if task == "segmentation":
            return self.seg_pipeline(data)
        elif task == "classification":
            if data.get("has_mask", True):
                return self.cls_pipeline(data)
            else:
                return self.cls_pipeline_no_mask(data)
        else:
            raise ValueError(f"Tâche inconnue : {task}")
