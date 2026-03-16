"""
Ablation model variants for AD-SAM.

Variants:
- full: Standard DualEncoderDeformableDecoder (no changes)
- no_deform: Replace deformable convolutions with standard convolutions
- no_attention: Remove channel attention from the fusion layer
- sam_encoder_only: SAM encoder only (simple conv decoder), no ResNet branch
- ce_loss: Full model but trained with CrossEntropyLoss (handled in train.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from models.deformable import DeformableConv2d, DeformableFeatureFusion
from models.attention import ChannelAttention


# ── Standard (non-deformable) replacements ──────────────────────────────────

class StandardConv2d(nn.Module):
    """Drop-in replacement for DeformableConv2d using a regular Conv2d."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        return self.conv(x)


class StandardFeatureFusion(nn.Module):
    """Fusion layer using standard convolutions, no deformable conv, no channel attention."""
    def __init__(self, sam_channels, resnet_channels):
        super().__init__()
        self.resnet_conv = nn.Conv2d(resnet_channels, sam_channels, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sam_channels * 2, sam_channels, 3, padding=1),
            nn.GroupNorm(8, sam_channels),
            nn.GELU(),
        )

    def forward(self, sam_feat, resnet_feat):
        resnet_feat = self.resnet_conv(resnet_feat)
        combined = torch.cat([sam_feat, resnet_feat], dim=1)
        return self.fusion_conv(combined)


class NoDeformFeatureFusion(nn.Module):
    """Fusion layer using standard convolutions but WITH channel attention."""
    def __init__(self, sam_channels, resnet_channels):
        super().__init__()
        self.resnet_conv = nn.Conv2d(resnet_channels, sam_channels, 1)
        self.fusion_conv = nn.Conv2d(sam_channels * 2, sam_channels, 3, padding=1)
        self.channel_attention = ChannelAttention(sam_channels)
        self.norm = nn.GroupNorm(8, sam_channels)
        self.activation = nn.GELU()

    def forward(self, sam_feat, resnet_feat):
        resnet_feat = self.resnet_conv(resnet_feat)
        combined = torch.cat([sam_feat, resnet_feat], dim=1)
        fused = self.fusion_conv(combined)
        fused = self.channel_attention(fused)
        return self.activation(self.norm(fused))


class NoAttentionFeatureFusion(nn.Module):
    """Deformable fusion WITHOUT channel attention."""
    def __init__(self, sam_channels, resnet_channels):
        super().__init__()
        self.resnet_conv = nn.Conv2d(resnet_channels, sam_channels, 1)
        self.fusion_conv = DeformableConv2d(sam_channels * 2, sam_channels,
                                            kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, sam_channels)
        self.activation = nn.GELU()

    def forward(self, sam_feat, resnet_feat):
        resnet_feat = self.resnet_conv(resnet_feat)
        combined = torch.cat([sam_feat, resnet_feat], dim=1)
        fused = self.fusion_conv(combined)
        return self.activation(self.norm(fused))


# ── Standard (non-deformable) upsample block ────────────────────────────────

class StandardUpsampleBlock(nn.Module):
    """UpsampleBlock using standard Conv2d instead of DeformableConv2d."""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.skip_channels = skip_channels

        if skip_channels > 0:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, skip_channels // 4, 1),
                nn.GroupNorm(8, skip_channels // 4),
                nn.GELU()
            )
            total_in = in_channels + skip_channels // 4
        else:
            total_in = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None and self.skip_channels > 0:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear',
                                     align_corners=False)
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── SAM-only decoder (no ResNet branch) ─────────────────────────────────────

class SAMOnlyDecoder(nn.Module):
    """Uses only SAM encoder features; simple conv decoder from 64x64 to 1024x1024."""
    def __init__(self, sam_model, num_classes=19):
        super().__init__()
        self.sam_encoder = sam_model.image_encoder

        # SAM normalization buffers (convert ImageNet-normalized input to SAM scale)
        self.register_buffer('_imgnet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_imgnet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer('_sam_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('_sam_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

        # Simple progressive decoder from SAM's 256-ch 64x64 output
        self.up1 = self._make_up_block(256, 128)    # 64 -> 128
        self.up2 = self._make_up_block(128, 64)     # 128 -> 256
        self.up3 = self._make_up_block(64, 32)      # 256 -> 512
        self.up4 = self._make_up_block(32, 32)      # 512 -> 1024

        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, 1)
        )

    @staticmethod
    def _make_up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x, sam_embedding=None):
        B, C, H, W = x.shape
        if (H, W) != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        if sam_embedding is not None:
            sam_features = sam_embedding
        else:
            # Re-normalize from ImageNet to SAM scale
            x_sam = (x * self._imgnet_std + self._imgnet_mean) * 255.0
            x_sam = (x_sam - self._sam_mean) / self._sam_std
            sam_features = self.sam_encoder(x_sam)  # [B, 256, 64, 64]

        x = F.interpolate(sam_features, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)     # -> [B, 128, 128, 128]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)     # -> [B, 64, 256, 256]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x)     # -> [B, 32, 512, 512]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x)     # -> [B, 32, 1024, 1024]

        logits = self.classifier(x)

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits


# ── Builder function ─────────────────────────────────────────────────────────

def build_ablation_model(ablation, sam_model, num_classes=19):
    """
    Build model variant for ablation study.

    Args:
        ablation: one of "full", "no_deform", "no_attention", "sam_encoder_only", "ce_loss"
        sam_model: initialized SAM model
        num_classes: number of segmentation classes

    Returns:
        nn.Module: the constructed model
    """
    from models.dual_encoder import DualEncoderDeformableDecoder

    if ablation in ("full", "ce_loss"):
        # ce_loss uses the full architecture, just a different loss function
        return DualEncoderDeformableDecoder(sam_model, num_classes=num_classes)

    elif ablation == "no_deform":
        model = DualEncoderDeformableDecoder(sam_model, num_classes=num_classes)
        # Replace fusion layer: standard conv but KEEP channel attention
        model.fusion_layers = nn.ModuleList([
            NoDeformFeatureFusion(256, 2048)
        ])
        # Replace deformable upsample blocks with standard conv blocks
        model.up1 = StandardUpsampleBlock(256, 128, skip_channels=1024)
        model.up2 = StandardUpsampleBlock(128, 64, skip_channels=512)
        model.up3 = StandardUpsampleBlock(64, 32, skip_channels=256)
        model.up4 = StandardUpsampleBlock(32, 32, skip_channels=0)
        return model

    elif ablation == "no_attention":
        model = DualEncoderDeformableDecoder(sam_model, num_classes=num_classes)
        # Replace fusion layer: keep deformable conv, remove channel attention
        model.fusion_layers = nn.ModuleList([
            NoAttentionFeatureFusion(256, 2048)
        ])
        return model

    elif ablation == "sam_encoder_only":
        return SAMOnlyDecoder(sam_model, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown ablation: {ablation}")
