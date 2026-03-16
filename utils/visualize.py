import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils.crf_postprocess import crf_postprocess

def visualize_predictions_dual(
    model,
    val_loader,
    epoch,
    save_dir="dual-encoder-visualizations",
    num_samples=4,
    apply_crf=False
):
    os.makedirs(save_dir, exist_ok=True)
    
    color_map = {
        0: (128, 64, 128),   # road
        1: (244, 35, 232),   # sidewalk
        2: (70, 70, 70),     # building
        3: (102, 102, 156),  # wall
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        6: (250, 170, 30),   # traffic light
        7: (220, 220, 0),    # traffic sign
        8: (107, 142, 35),   # vegetation
        9: (152, 251, 152),  # terrain
        10: (70, 130, 180),  # sky
        11: (220, 20, 60),   # person
        12: (255, 0, 0),     # rider
        13: (0, 0, 142),     # car
        14: (0, 0, 70),      # truck
        15: (0, 60, 100),    # bus
        16: (0, 80, 100),    # train
        17: (0, 0, 230),     # motorcycle
        18: (119, 11, 32),   # bicycle
        19: (0, 0, 0)        # ignore/background
    }

    def create_color_mask(mask):
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in color_map.items():
            color_mask[mask == label] = color
        return color_mask

    model.eval()
    
    with torch.no_grad():
        for i, (batch_input, target_masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            device = next(model.parameters()).device
            images = batch_input["image"].to(device)
            target_masks = target_masks.to(device)
            sam_emb = batch_input.get("sam_embedding")
            if sam_emb is not None:
                sam_emb = sam_emb.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images, sam_embedding=sam_emb)
            if logits.shape[-2:] != target_masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=target_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Decide prediction
            if apply_crf:
                # compute per-class probabilities [B, C, H, W]
                probs = torch.softmax(logits, dim=1)
                # take first sample
                prob_tensor = probs[0]  # [C, H, W]
                
                # prepare image for CRF: denormalize and to uint8 HxWx3
                inp = images[0].cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img_np = (inp.transpose(1,2,0) * std + mean) * 255
                img_np = img_np.astype(np.uint8)
                
                # get numpy probabilities in [C, H, W]
                prob_np = prob_tensor.cpu().numpy()
                
                # CRF refine
                pred_mask = crf_postprocess(
                    img_np,
                    prob_np,
                    n_classes=prob_np.shape[0],
                    iter_max=10
                )
            else:
                pred_masks = torch.argmax(logits, dim=1)
                pred_mask = pred_masks[0].cpu().numpy()
            
            # Convert to numpy for visualization
            input_image = images[0].cpu().numpy()
            target_mask = target_masks[0].cpu().numpy()
            
            # Denormalize input image for display
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            input_image = (input_image.transpose(1, 2, 0) * std + mean) * 255
            input_image = input_image.astype(np.uint8)
            
            # Colorize masks
            target_color = create_color_mask(target_mask)
            pred_color   = create_color_mask(pred_mask)
            
            # Plot
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(target_color)
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            title = 'Prediction' + (' (CRF)' if apply_crf else '')
            plt.imshow(pred_color)
            plt.title(title)
            plt.axis('off')
            
            # Save
            save_path = os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
