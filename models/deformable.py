import torch
import torch.nn as nn
import torchvision
from .attention import ChannelAttention

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(DeformableConv2d, self).__init__()
        
        self.conv = torchvision.ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        
        # Offset prediction conv (2 * kernel_size * kernel_size offsets for each position)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Modulation (mask) prediction conv (kernel_size * kernel_size masks for each position)
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.conv(x, offset, mask)


class DeformableFeatureFusion(nn.Module):
    def __init__(self, sam_channels, resnet_channels):
        super().__init__()
        
        # Reduce ResNet channels to match SAM
        self.resnet_conv = nn.Conv2d(resnet_channels, sam_channels, 1)
        
        # Deformable convolution for feature fusion
        self.fusion_conv = DeformableConv2d(
            sam_channels * 2,  # Concatenated features
            sam_channels,
            kernel_size=3,
            padding=1
        )
        
        # Channel attention
        self.channel_attention = ChannelAttention(sam_channels)
        
        # Additional normalization and activation
        self.norm = nn.GroupNorm(8, sam_channels)
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, sam_feat, resnet_feat):
        # Process ResNet features - input channels match resnet_channels
        resnet_feat = self.resnet_conv(resnet_feat)  # Output: sam_channels
        
        # Concatenate features
        combined = torch.cat([sam_feat, resnet_feat], dim=1)
        
        # Fuse using deformable convolution
        fused = self.fusion_conv(combined)
        
        # Apply attention and normalization
        fused = self.channel_attention(fused)
        fused = self.activation(self.norm(fused))
        
        return fused