import monai.transforms as T

def get_train_transforms():
    return T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,a_max=140, 
                b_min=0.0, b_max=1.0, 
                clip=True),
            T.RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            T.RandRotate90d(keys=["image"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            T.ToTensord(keys=["image", "label"])])  
    
def get_val_transforms():
    return  T.Compose([
            T.LoadImaged(keys=["image"], image_only=True),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes='RAS'),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            T.CropForegroundd(keys=["image"], source_key='image'),
            T.ScaleIntensityRanged(
                keys=["image"],
                a_min=-10,a_max=140, 
                b_min=0.0, b_max=1.0, 
                clip=True),
            T.ToTensord(keys=["image", "label"]) ])