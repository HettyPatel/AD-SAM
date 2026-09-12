import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 dice_weight=0.5,
                 ignore_index=19,
                 num_classes=19):
        
        """
        Combined Focal and Dice Loss for Semantic Segmentation

        Args:
            alpha (float): Weighting factor for Focal Loss
            gamma (float): Focusing parameter for Focal Loss
            dice_weight (float): Weighting factor for Dice Loss
            ignore_index (int): Index to ignore in loss computation
            num_classes (int): Number of classes in the dataset
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def focal_loss(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Shape [B, C, H, W]
            targets (torch.Tensor): Shape [B, H, W]
        """
        # Ensure inputs are in correct shape [B, C, H, W]
        assert len(inputs.shape) == 4, f"Expected inputs to be [B,C,H,W], got {inputs.shape}"
        assert len(targets.shape) == 3, f"Expected targets to be [B,H,W], got {targets.shape}"
        
        # Convert inputs to probabilities with log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)  # [B, C, H, W]
        probs = torch.exp(log_probs)  # [B, C, H, W]
        
        # Create mask for valid pixels (not ignore_index)
        mask = (targets != self.ignore_index).float()  # [B, H, W]
        
        # Get target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1).clamp(0, self.num_classes-1))  # [B, 1, H, W]
        target_probs = target_probs.squeeze(1)  # [B, H, W]
        
        # Compute focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * log_probs.gather(1, targets.unsqueeze(1).clamp(0, self.num_classes-1)).squeeze(1)
        
        # Apply mask and take mean
        focal_loss = (focal_loss * mask).sum() / (mask.sum() + 1e-7)
        
        return focal_loss

    def dice_loss(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Shape [B, C, H, W]
            targets (torch.Tensor): Shape [B, H, W]
        """
        # Ensure inputs are in correct shape [B, C, H, W]
        assert len(inputs.shape) == 4, f"Expected inputs to be [B,C,H,W], got {inputs.shape}"
        assert len(targets.shape) == 3, f"Expected targets to be [B,H,W], got {targets.shape}"
        
        # Convert inputs to probabilities
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # Create mask for valid pixels (not ignore_index)
        mask = (targets != self.ignore_index).float()  # [B, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
        # One-hot encode targets with ignore_index handling
        valid_targets = targets * (targets != self.ignore_index).long()  # Zero out ignored indices
        targets_one_hot = F.one_hot(valid_targets, num_classes=self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Apply mask
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask
        
        # Compute intersection and union per class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # [B, C]
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))  # [B, C]
        
        # Compute Dice score
        dice_score = (2. * intersection + 1e-7) / (union + 1e-7)  # [B, C]
        
        # Compute mean Dice score across all valid classes and batches
        valid_pixels = mask.sum(dim=(2, 3)) > 0  # [B, 1]
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss
    
    def forward(self, inputs, targets):
        """
        Compute the combined Focal and Dice Loss
        
        Args:
            inputs (torch.Tensor): Network predictions (Logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined Focal and Dice Loss
        """
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
