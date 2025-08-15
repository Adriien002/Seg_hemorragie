from monai import transforms as T



class TaskBasedTransform:
    """
    Applique un pipeline différent selon la tâche : "segmentation" ou "classification".
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedTransform initialized")

        self.window_preset = {"window_center": 40, "window_width": 80}

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
                spatial_size=(96, 96, 96),
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
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144)), #bof ( trop petit) pour classification mais on test
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            T.ToTensord(keys=["image", "label"])
        ])
        
    def __call__(self, data):
        task = data["task"]
        if task == "segmentation":
            return self.seg_pipeline(data)
        elif task == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {task}")
        
# def get_multitask_transforms():
#     return TaskBasedTransform(keys=["image", "label"])
   

        
class TaskBasedValTransform:
    """
    Transformations de validation — une pipeline par tâche, sans augmentation aléatoire.
    """
    def __init__(self, keys):
        
        print(">>> TaskBasedTransform initialized")
        self.window_preset = {"window_center": 40, "window_width": 80}
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
            T.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
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
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 144)),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,
                a_max=140,
                b_min=0.0, b_max=1.0, clip=True
            ),
            T.ToTensord(keys=["image", "label"])
        ])


    def __call__(self, data):
        print("Pipeline called for task:", data["task"])
        if data["task"] == "segmentation":
            return self.seg_pipeline(data)
        elif data["task"] == "classification":
            return self.cls_pipeline(data)
        else:
            raise ValueError(f"Tâche inconnue : {data['task']}")