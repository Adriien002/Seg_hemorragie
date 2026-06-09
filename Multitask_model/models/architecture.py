from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F



class BasicUNetWithClassification(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels_orig: int = 6,      # Classes pour RSNA/orig
        out_channels_inhouse: int = 4,   # Classes pour in-house
        out_channels_instance: int = 2,  # Classes pour INSTANCE 2022
        num_cls_classes: int = 6,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # ==========================================
        # 1. ENCODEUR (Partagé par toutes les tâches)
        # ==========================================
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        
        # ==========================================
        # 2. DÉCODEUR A : Segmentation Originale (6 classes)
        # ==========================================
        self.upcat_4_orig = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3_orig = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2_orig = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1_orig = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv_orig = Conv["conv", spatial_dims](fea[5], out_channels_orig, kernel_size=1)

        # ==========================================
        # 3. DÉCODEUR B : Segmentation In-House (4 classes)
        # ==========================================
        self.upcat_4_inhouse = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3_inhouse = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2_inhouse = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1_inhouse = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv_inhouse = Conv["conv", spatial_dims](fea[5], out_channels_inhouse, kernel_size=1)

        # ==========================================
        # 4. DÉCODEUR C : Segmentation INSTANCE 2022 (2 classes)
        # ==========================================
        self.upcat_4_instance = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3_instance = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2_instance = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1_instance = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv_instance = Conv["conv", spatial_dims](fea[5], out_channels_instance, kernel_size=1)
     
        # ==========================================
        # 4. TÊTE C : Classification
        # ==========================================
        self.cls_head = nn.Sequential(
            nn.Conv3d(fea[4], fea[4], kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.2),  # dropout spatial sur les feature maps
            nn.AdaptiveMaxPool3d((2,2,2)),
            nn.Flatten(),
            nn.Linear(fea[4] * 8, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls_classes)
        )

        # Projection spatiale bottleneck → espace classes pour Lmta
        self.cls_spatial_proj = Conv["conv", spatial_dims](fea[4], out_channels_orig, kernel_size=1)
   
    def forward(self, x: torch.Tensor, task: str = "seg_orig", batch_size: int = None):
        # Encodeur commun
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        # cls_head seulement pour les tâches qui utilisent cls_logits en loss
        cls_logits = self.cls_head(x4) if task not in ("seg_inhouse", "seg_instance") else None

        if task == "seg_inhouse":
            u4 = self.upcat_4_inhouse(x4, x3)
            u3 = self.upcat_3_inhouse(u4, x2)
            u2 = self.upcat_2_inhouse(u3, x1)
            u1 = self.upcat_1_inhouse(u2, x0)
            seg_logits = self.final_conv_inhouse(u1)
            return seg_logits, None, None, None

        if task == "seg_instance":
            u4 = self.upcat_4_instance(x4, x3)
            u3 = self.upcat_3_instance(u4, x2)
            u2 = self.upcat_2_instance(u3, x1)
            u1 = self.upcat_1_instance(u2, x0)
            seg_logits = self.final_conv_instance(u1)
            return seg_logits, None, None, None

        # Décodeur seg_orig — aussi utilisé pour classification (P/S pour Lmta)
        u4 = self.upcat_4_orig(x4, x3)
        u3 = self.upcat_3_orig(u4, x2)
        u2 = self.upcat_2_orig(u3, x1)
        u1 = self.upcat_1_orig(u2, x0)
        seg_logits = self.final_conv_orig(u1)
        S = torch.softmax(seg_logits, dim=1)
        P = torch.softmax(self.cls_spatial_proj(x4), dim=1)

        return seg_logits, cls_logits, P, S