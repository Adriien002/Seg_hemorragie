import monai.transforms as T


def get_train_transforms(cfg):
    return T.Compose([
        # Loading transforms
        T.LoadImaged(keys=["image", "seg"]),
        T.EnsureChannelFirstd(keys=["image", "seg"]),
        T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
        T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
        T.Spacingd(keys=["image", "seg"],pixdim=(1., 1., 1.),mode=['bilinear', 'nearest']),
        T.SpatialPadd(keys=["image", "seg"],spatial_size=cfg["augmentation"]["spatial_size"]),
        T.ScaleIntensityRanged(keys=["image"],a_min=-10, a_max=140,b_min=0.0, b_max=1.0,clip=True),

        # Random crops
        T.RandCropByPosNegLabeld(
            keys=['image', 'seg'],
            image_key='image',
            label_key='seg',
            pos=cfg["augmentation"]["pos_ratio"],
            neg=cfg["augmentation"]["neg_ratio"],
            spatial_size=cfg["augmentation"]["spatial_size"],
            num_samples=2
        ),

        #  Data augmentations
            T.RandFlipd(keys=["image", "seg"], spatial_axis=[0, 1], prob=0.5),
            T.RandRotate90d(keys=["image", "seg"], spatial_axes=(0, 1), prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=0.02, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5)
    ])


def get_val_transforms(cfg):
    return T.Compose([
        T.LoadImaged(keys=["image", "seg"]),
        T.EnsureChannelFirstd(keys=["image", "seg"]),
        T.CropForegroundd(keys=['image', 'seg'], source_key='image'),
        T.Orientationd(keys=["image", "seg"], axcodes='RAS'),
        T.Spacingd(keys=["image", "seg"],
                   pixdim=(1., 1., 1.),
                   mode=['bilinear', 'nearest']),
        T.SpatialPadd(keys=["image", "seg"],
                      spatial_size=cfg["augmentation"]["spatial_size"]),
        T.ScaleIntensityRanged(keys=["image"],
                               a_min=-10, a_max=140,
                               b_min=0.0, b_max=1.0,
                               clip=True),
    ])