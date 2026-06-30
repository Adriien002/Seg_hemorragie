import math
import monai.transforms as T
import config


def get_transforms(subset: str, has_mask: bool):
    """Pipeline classification uniquement (image + class_label).
    has_mask=True  : données positives avec pseudo-masque → crop guidé sur la lésion.
    has_mask=False : données saines sans masque → crop spatial aléatoire.
    """
    keys = ["image", "label"] if has_mask else ["image"]
    seg_mode = ["trilinear", "nearest"] if has_mask else "trilinear"

    transforms = [
        T.LoadImaged(keys=keys, image_only=True),
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=seg_mode),
        T.CropForegroundd(keys=keys, source_key="image"),
        T.ScaleIntensityRanged(keys=["image"], a_min=-10, a_max=140, b_min=0.0, b_max=1.0, clip=True),
        T.SpatialPadd(keys=keys, spatial_size=config.img_size, mode="constant", value=0.0),
        T.NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=config.non_zero_norm),
    ]

    if subset == "train":
        transforms += [
            T.RandAffined(
                keys=keys, prob=0.5, rotate_range=math.radians(30), scale_range=0.1,
                mode=["bilinear", "nearest"] if has_mask else "bilinear", padding_mode="border",
            ),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            T.RandFlipd(keys=keys, spatial_axis=[0, 1], prob=0.5),
        ]
        if has_mask:
            transforms.append(
                T.RandCropByPosNegLabeld(
                    keys=["image", "label"], image_key="image", label_key="label",
                    pos=5.0, neg=1.0, spatial_size=config.img_size, num_samples=2,
                )
            )
        else:
            transforms.append(
                T.RandSpatialCropSamplesd(
                    keys=["image"], roi_size=config.img_size, num_samples=1, random_size=False,
                )
            )
    else:  # validation
        if has_mask:
            transforms.append(
                T.RandCropByPosNegLabeld(
                    keys=["image", "label"], image_key="image", label_key="label",
                    pos=1.0, neg=0.0, spatial_size=config.img_size, num_samples=1,
                )
            )
        else:
            transforms.append(
                T.RandSpatialCropSamplesd(
                    keys=["image"], roi_size=config.img_size, num_samples=1, random_size=False,
                )
            )

    if has_mask:
        transforms.append(T.DeleteItemsd(keys=["label"]))

    transforms.append(T.ToTensord(keys=["image", "class_label"]))
    return T.Compose(transforms)
