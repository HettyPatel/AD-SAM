import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance_transform

# Lovász-Softmax loss

def lovasz_grad(gt_sorted):
    """Gradient of Lovász extension w.r.t sorted errors"""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas, labels, ignore=None):
    """Flatten predictions: [B,C,H,W] -> [P,C] and labels: [B,H,W] -> [P]"""
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]

def lovasz_softmax_flat(probas, labels, classes='present'):
    """Lovász-Softmax loss on flattened tensors"""
    if probas.numel() == 0:
        # no valid pixels
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == 'present' else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if losses:
        return sum(losses) / len(losses)
    return torch.tensor(0.0, device=probas.device)

def lovasz_softmax(probas, labels, classes='present', per_image=True, ignore=None):
    if per_image:
        loss = 0.0
        for prob, lab in zip(probas, labels):
            loss += lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
        loss /= probas.size(0)
        return loss
    else:
        probas, labels = flatten_probas(probas, labels, ignore)
        return lovasz_softmax_flat(probas, labels, classes=classes)

class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index=19, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1)
        loss = lovasz_softmax(probas, targets, per_image=self.per_image, ignore=self.ignore_index)
        return loss

# Surface loss

class SurfaceLoss(nn.Module):
    def __init__(self, ignore_index=19):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """Weight per-pixel error by distance to nearest boundary"""
        probs = F.softmax(inputs, dim=1)
        B, C, H, W = inputs.shape
        loss = 0.0

        for b in range(B):
            target_np = targets[b].detach().cpu().numpy().astype(np.int32)
            
            # detect boundaries by comparing neighbors
            boundaries = np.zeros_like(target_np, dtype=bool)
            boundaries[1:, :] |= (target_np[1:, :] != target_np[:-1, :])
            boundaries[:-1, :] |= (target_np[:-1, :] != target_np[1:, :])
            boundaries[:, 1:] |= (target_np[:, 1:] != target_np[:, :-1])
            boundaries[:, :-1] |= (target_np[:, :-1] != target_np[:, 1:])
            
            # distance to nearest boundary
            distance_map = distance_transform(~boundaries)
            distance_map = torch.from_numpy(distance_map).to(inputs.device).float()
            distance_map = distance_map.unsqueeze(0)
            
            valid_mask = (targets[b] != self.ignore_index)
            target_one_hot = F.one_hot(targets[b].clamp(0, C - 1), num_classes=C).permute(2, 0, 1).float()
            valid_mask = valid_mask.unsqueeze(0)
            
            diff = torch.abs(probs[b] - target_one_hot)
            weighted_error = diff * distance_map * valid_mask
            loss += weighted_error.mean()
        loss = loss / B
        return loss

# Focal and Dice losses

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=19, num_classes=19):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        assert inputs.ndim == 4, f"Expected inputs of shape [B,C,H,W], got {inputs.shape}"
        assert targets.ndim == 3, f"Expected targets of shape [B,H,W], got {targets.shape}"
        
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        mask = (targets != self.ignore_index).float()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1).clamp(0, self.num_classes - 1))
        target_probs = probs.gather(1, targets.unsqueeze(1).clamp(0, self.num_classes - 1))
        target_log_probs = target_log_probs.squeeze(1)
        target_probs = target_probs.squeeze(1)
        
        focal_weight = (1 - target_probs) ** self.gamma
        loss = -self.alpha * focal_weight * target_log_probs
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-7)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=19, num_classes=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        assert inputs.ndim == 4, f"Expected inputs of shape [B,C,H,W], got {inputs.shape}"
        assert targets.ndim == 3, f"Expected targets of shape [B,H,W], got {targets.shape}"
        
        probs = F.softmax(inputs, dim=1)
        mask = (targets != self.ignore_index).float().unsqueeze(1)
        
        valid_targets = targets * (targets != self.ignore_index).long()
        targets_one_hot = F.one_hot(valid_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

# Hybrid loss: 40% Focal, 30% Dice, 20% Lovász, 10% Surface

class HybridLoss(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.focal = FocalLoss(num_classes=num_classes)
        self.dice = DiceLoss(num_classes=num_classes)
        self.lovasz = LovaszSoftmax(ignore_index=19, per_image=True)
        self.boundary = SurfaceLoss(ignore_index=19)
        
    def forward(self, inputs, targets):
        loss = (0.4 * self.focal(inputs, targets) +
                0.3 * self.dice(inputs, targets) +
                0.2 * self.lovasz(inputs, targets) +
                0.1 * self.boundary(inputs, targets))
        return loss
