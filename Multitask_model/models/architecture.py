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
        decoder_sharing: str = "last_block",  # "none" | "last_block" (B) | "full" (A)
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()

        if decoder_sharing not in ("none", "last_block", "full"):
            raise ValueError(f"decoder_sharing invalide: {decoder_sharing!r}")
        self.decoder_sharing = decoder_sharing

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea} | decoder_sharing: {decoder_sharing}.")

        # ==========================================
        # 1. ENCODEUR (Partagé par toutes les tâches)
        # ==========================================
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # --- fabriques de blocs décodeur (réutilisent fea/act/norm/...) ---
        def _full_decoder(out_channels: int) -> nn.ModuleDict:
            """Décodeur U-Net complet (4 UpCat + final conv 1x1)."""
            return nn.ModuleDict({
                "upcat_4": UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample),
                "upcat_3": UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample),
                "upcat_2": UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample),
                "upcat_1": UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False),
                "final_conv": Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1),
            })

        def _last_block_head(out_channels: int) -> nn.ModuleDict:
            """Tête variante B : dernier UpCat (résolution fine) + final conv 1x1."""
            return nn.ModuleDict({
                "upcat_1": UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False),
                "final_conv": Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1),
            })

        def _final_head(out_channels: int) -> nn.Module:
            """Tête variante A : seulement un final conv 1x1."""
            return Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        # ==========================================
        # 2. DÉCODEUR(S) selon le mode de partage
        # ==========================================
        if decoder_sharing == "none":
            # 3 décodeurs complets indépendants (archi historique)
            self.dec_orig = _full_decoder(out_channels_orig)
            self.dec_inhouse = _full_decoder(out_channels_inhouse)
            self.dec_instance = _full_decoder(out_channels_instance)
        else:
            # Tronc de décodeur PARTAGÉ (upcat_4 -> upcat_2) commun aux 3 datasets
            self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
            self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
            self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)

            if decoder_sharing == "last_block":  # Variante B
                self.head_orig = _last_block_head(out_channels_orig)
                self.head_inhouse = _last_block_head(out_channels_inhouse)
                self.head_instance = _last_block_head(out_channels_instance)
            else:  # "full" -> Variante A : dernier UpCat partagé aussi
                self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
                self.head_orig = _final_head(out_channels_orig)
                self.head_inhouse = _final_head(out_channels_inhouse)
                self.head_instance = _final_head(out_channels_instance)

        # ==========================================
        # 3. TÊTE : Classification (sur le bottleneck)
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

    def _decode(self, task, x0, x1, x2, x3, x4):
        """Reconstruit la carte de segmentation selon le mode de partage et la tâche."""
        if self.decoder_sharing == "none":
            dec = {
                "seg_inhouse": self.dec_inhouse,
                "seg_instance": self.dec_instance,
            }.get(task, self.dec_orig)
            u4 = dec["upcat_4"](x4, x3)
            u3 = dec["upcat_3"](u4, x2)
            u2 = dec["upcat_2"](u3, x1)
            u1 = dec["upcat_1"](u2, x0)
            return dec["final_conv"](u1)

        # Tronc partagé (variantes A et B)
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)

        head = {
            "seg_inhouse": self.head_inhouse,
            "seg_instance": self.head_instance,
        }.get(task, self.head_orig)

        if self.decoder_sharing == "full":  # Variante A : dernier UpCat partagé
            u1 = self.upcat_1(u2, x0)
            return head(u1)

        # Variante B : dernier UpCat propre à chaque dataset
        u1 = head["upcat_1"](u2, x0)
        return head["final_conv"](u1)

    def forward(self, x: torch.Tensor, task: str = "seg_orig", batch_size: int = None):
        # Encodeur commun
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        # cls_head seulement pour les tâches qui utilisent cls_logits en loss
        cls_logits = self.cls_head(x4) if task not in ("seg_inhouse", "seg_instance") else None

        seg_logits = self._decode(task, x0, x1, x2, x3, x4)

        if task in ("seg_inhouse", "seg_instance"):
            return seg_logits, None, None, None

        # Décodeur seg_orig — aussi utilisé pour classification (P/S pour Lmta)
        S = torch.softmax(seg_logits, dim=1)
        P = torch.softmax(self.cls_spatial_proj(x4), dim=1)

        return seg_logits, cls_logits, P, S

