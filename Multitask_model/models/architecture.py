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
        out_channels_orig: int = 6,     # Classes pour RSNA/orig
        out_channels_inhouse: int = 4,  # Classes pour in-house
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
   
    def forward(self, x: torch.Tensor, task: str = "seg_orig", batch_size: int = None):
        # --- PASSAGE DANS L'ENCODEUR COMMUN ---
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        
        # --- ROUTAGE VERS LA BONNE TÊTE ---
        if task == "classification":
            # On utilise le masque d'attention sur le décodeur "orig" (qui contient le plus de classes)
            # # pour guider la classification, comme tu le faisais avant.
            # u4 = self.upcat_4_orig(x4, x3)
            # u3 = self.upcat_3_orig(u4, x2)
            # u2 = self.upcat_2_orig(u3, x1)
            # u1 = self.upcat_1_orig(u2, x0)
            # seg_logits = self.final_conv_orig(u1)
            
            # probs = torch.softmax(seg_logits, dim=1)
            # lesion_attention = probs[:, 1:, ...].sum(dim=1, keepdim=True) 
            # pooled_attention = F.adaptive_max_pool3d(lesion_attention, output_size=x4.shape[2:])
            
            # masked_x4 = x4 * (1.0 + pooled_attention)
            cls_logits = self.cls_head(x4)
            return None, cls_logits

        elif task == "seg_orig":
            u4 = self.upcat_4_orig(x4, x3)
            u3 = self.upcat_3_orig(u4, x2)
            u2 = self.upcat_2_orig(u3, x1)
            u1 = self.upcat_1_orig(u2, x0)
            seg_logits = self.final_conv_orig(u1)
            return seg_logits, None
            
        elif task == "seg_inhouse":
            u4 = self.upcat_4_inhouse(x4, x3)
            u3 = self.upcat_3_inhouse(u4, x2)
            u2 = self.upcat_2_inhouse(u3, x1)
            u1 = self.upcat_1_inhouse(u2, x0)
            seg_logits = self.final_conv_inhouse(u1)
            return seg_logits, None
            
        else:
            raise ValueError(f"Tâche non reconnue dans l'architecture : {task}")