from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import TwoConv, Down
from monai.networks.nets import DenseNet121
from monai.utils import ensure_tuple_rep


class BasicUNetClassifier(nn.Module):
    """Encodeur BasicUNet + tête de classification uniquement.
    Même encodeur que le modèle multi-tâche pour comparaison en ablation.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_cls_classes: int = 6,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
    ):
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNetClassifier features: {fea}.")

        # Encodeur — identique au multi-tâche
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # Tête classification — identique au multi-tâche
        self.cls_head = nn.Sequential(
            nn.Conv3d(fea[4], fea[4], kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.AdaptiveMaxPool3d((2, 2, 2)),
            nn.Flatten(),
            nn.Linear(fea[4] * 8, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        return self.cls_head(x4)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 3D — baseline classification.
    Utilise l'implémentation MONAI (spatial_dims=3).
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_cls_classes: int = 6,
    ):
        super().__init__()
        self.model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_cls_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
