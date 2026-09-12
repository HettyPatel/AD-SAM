import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from .deformable import DeformableConv2d, DeformableFeatureFusion
from .attention import ChannelAttention

class DualEncoderDeformableDecoder(nn.Module):
    """
    Frozen SAM ViT-H (256ch, 64x64) + trainable ResNet-50, fused at 64x64 by a
    deformable convolution with channel attention, then a progressive deformable
    decoder with ResNet skip connections:
        up1 64->128 (+res3), up2 128->256 (+res2), up3 refine at 256 (+res1, no upsample),
        head at 256x256 (stride 4), logits bilinearly upsampled x4 to the input size.
    """
    def __init__(self, sam_model, num_classes=19):
        super().__init__()

        # SAM encoder (frozen; an Identity stub in cached-embedding mode)
        self.sam_encoder = sam_model.image_encoder

        # SAM normalization: convert ImageNet-normalized input to SAM-normalized
        # ImageNet: (pixel/255 - mean_in) / std_in  =>  pixel = (x * std_in + mean_in) * 255
        # SAM: (pixel - mean_sam) / std_sam
        self.register_buffer('_imgnet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_imgnet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer('_sam_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('_sam_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

        # ResNet encoder
        resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.resnet_encoder = create_feature_extractor(
            resnet,
            return_nodes={
                'layer1': 'res1',  # 256 ch, stride 4  -> 256x256
                'layer2': 'res2',  # 512 ch, stride 8  -> 128x128
                'layer3': 'res3',  # 1024 ch, stride 16 -> 64x64
                'layer4': 'res4'   # 2048 ch, stride 32 -> 32x32
            }
        )

        # Fusion at 64x64: SAM (256ch) + ResNet layer4 (2048ch, bilinearly upsampled 32->64)
        self.fusion_layers = nn.ModuleList([
            DeformableFeatureFusion(256, 2048),
        ])

        # Progressive decoder with ResNet skip connections
        self.up1 = UpsampleBlock(256, 128, skip_channels=1024)          # 64->128,  skip=res3
        self.up2 = UpsampleBlock(128, 64, skip_channels=512)            # 128->256, skip=res2
        self.up3 = UpsampleBlock(64, 32, skip_channels=256, upsample=False)  # refine at 256, skip=res1

        # Classification head at 256x256 (stride 4)
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x, sam_embedding=None):
        B, C, H, W = x.shape
        if (H, W) != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Encode
        if sam_embedding is not None:
            sam_features = sam_embedding  # pre-computed [B, 256, 64, 64]
        else:
            # Re-normalize from ImageNet to SAM scale for the encoder
            x_sam = (x * self._imgnet_std + self._imgnet_mean) * 255.0  # back to 0-255
            x_sam = (x_sam - self._sam_mean) / self._sam_std
            sam_features = self.sam_encoder(x_sam)  # [B, 256, 64, 64]
        resnet_features = self.resnet_encoder(x)

        # Fuse SAM + ResNet layer4 at 64x64
        res4_up = F.interpolate(
            resnet_features['res4'],
            size=sam_features.shape[-2:],
            mode='bilinear', align_corners=False
        )
        fused = self.fusion_layers[0](sam_features, res4_up)  # [B, 256, 64, 64]

        # Progressive decode with skip connections
        x = self.up1(fused, resnet_features['res3'])   # -> [B, 128, 128, 128]
        x = self.up2(x, resnet_features['res2'])        # -> [B, 64, 256, 256]
        x = self.up3(x, resnet_features['res1'])        # -> [B, 32, 256, 256]

        logits = self.classifier(x)                     # [B, num_classes, 256, 256]

        if logits.shape[-2:] != (H, W):                 # bilinear x4 to the input size
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits


class UpsampleBlock(nn.Module):
    """[2x bilinear upsample] -> concat 1x1-reduced skip -> two deformable convs (GN + GELU).

    upsample=False turns the block into a same-resolution refinement block (used as the
    final stride-4 stage so res1 is concatenated at its native 256x256).
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, upsample=True):
        super().__init__()
        self.skip_channels = skip_channels
        self.upsample = upsample

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
            DeformableConv2d(total_in, out_channels),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            DeformableConv2d(out_channels, out_channels),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if skip is not None and self.skip_channels > 0:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)
