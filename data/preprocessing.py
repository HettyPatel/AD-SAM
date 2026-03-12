import torch

def preprocess_batch(batch_input):
    """Preprocess batch for model input"""
    # Input = float and normalized to [0, 1]
    
    # Resize to 1024x1024 (SAM's expected input size)
    batch_input["image"] = torch.nn.functional.interpolate(
        batch_input["image"],
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    )
    
    # Add ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch_input["image"].device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch_input["image"].device)
    batch_input["image"] = (batch_input["image"] - mean) / std
    
    return batch_input