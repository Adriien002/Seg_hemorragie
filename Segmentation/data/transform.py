
import os
import monai.transforms as T

transforms = T.Compose([
    # Loading transforms
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'), # who cares about the background ?
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),  # make sure all images are the same orientation
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']), # to isotropic spacing
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),  # make sure we have at least 96 slices
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),  # clip images

    # Let's crop 2 patches per case using positive negative instead of class by class
    T.RandCropByPosNegLabeld(
       keys=['image', 'seg'],
       image_key='image',
       label_key='seg',
        pos=5.0,
        neg=1.0,
        spatial_size=(96, 96, 96),
        num_samples=2
    ),
    # T.RandCropByLabelClassesd(
    #     keys=["image", "seg"],
    #     label_key="seg",
    #     spatial_size=(128, 128, 64),  # Taille adaptée aux EDH/SDH
    #     num_classes=6,
    #     ratios=[0.1, 0.25, 0.15, 0.1, 0.2, 0.2],  # Priorité EDH(1), SAH(2)
    #     num_samples=4
    # ),
    # T.RandCropByLabelClassesd(
    # keys=["image", "seg"],
    # label_key="seg",
    # spatial_size=(96, 96, 96),  
    # num_classes=6,
    # ratios=[0.1, 0.3, 0.1, 0.1, 0.1, 0.1],  # + de poids pour les classes rares (classe 1 ici à 0.3 par ex)
    # num_samples=4, 
    # ),

    # Data augmentations
    # For intensity augmentations, small random transforms but often
    # For spatial augmentations, only along the sagittal and coronal axis
    T.RandScaleIntensityd(
        keys=['image'],
        factors=0.02,
        prob=0.5
    ),
    T.RandShiftIntensityd(
       keys=['image'],
        offsets=0.05,
        prob=0.5
    ),
    T.RandRotate90d(
        keys=['image', 'seg'],
        prob=0.5,
        max_k=2,
        spatial_axes=(0, 1)
    ),
    T.RandFlipd(
        keys=['image', 'seg'],
        prob=0.5,
        spatial_axis=[0, 1]
    )
    # T.RandAffined(
    # keys=['image', 'seg'],
    # prob=0.3,
    # rotate_range=(0, 0, 0.1),  # Petite rotation seulement en axial (Z)
    # scale_range=(0.1, 0.1, 0),  # Léger zoom dans le plan axial
    # mode=['bilinear', 'nearest'],
    # padding_mode='border'
    # ),

    # T.RandAdjustContrastd(
    # keys=['image'],
    # gamma=(0.7, 1.5),  # Gamme plus large pour capturer EDH subtils
    # prob=0.5
    # )
])

val_transforms = T.Compose([
    # Loading transforms
    T.LoadImaged(keys=["image", "seg"]),
    T.EnsureChannelFirstd(keys=["image", "seg"]),
    T.CropForegroundd(keys=['image', 'seg'], source_key='image'), # who cares about the background ?
    T.Orientationd(keys=["image", "seg"], axcodes='RAS'),  # make sure all images are the same orientation
    T.Spacingd(keys=["image", "seg"], pixdim=(1., 1., 1.), mode=['bilinear', 'nearest']), # to isotropic spacing
    T.SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),  # make sure we have at least 96 slices
    T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),  # clip images
])
