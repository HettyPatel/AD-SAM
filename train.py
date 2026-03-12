import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from models.dual_encoder import DualEncoderDeformableDecoder
from models.sam_adapter import MulticlassSam
from data.dataloaders import get_dataloader
from data.preprocessing import preprocess_batch
from losses.focal_dice import FocalDiceLoss
from losses.hybrid_loss import HybridLoss
from metrics.iou_metric import IoUMetric
from utils.visualize import visualize_predictions_dual
from configs.paths import *
from utils.setup import *
from datetime import datetime
from configs.paths import SAM_CHECKPOINT_PATH
from configs.paths import (CITYSCAPES_TRAIN_IMAGES, CITYSCAPES_TRAIN_MASKS,
                           CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_MASKS,
                           BDD100K_TRAIN_IMAGES, BDD100K_TRAIN_MASKS,
                           BDD100K_VAL_IMAGES, BDD100K_VAL_MASKS)
from utils.crf_postprocess import crf_postprocess
import cv2
import os
import sys
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TeeLogger:
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return self.terminal.isatty()
    
    def close(self):
        self.log.close()
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--max_samples', type=int, default=100, help='max num of samples')
parser.add_argument('--dataset_name', type=str,choices=['cityscapes', 'bdd100k'], default='cityscapes', help='Dataset to use: "cityscapes" or "bdd100k"')
parser.add_argument('--num_epochs', type=int, default='100', help='max epochs')
parser.add_argument("--apply_crf",action="store_true",default=False,help="apply CRF post-processing on network output")
parser.add_argument("--loss",type=str, choices=['hybrid', 'focaldice'], default='hybrid', help='choose loss function')
parser.add_argument("--gpu",type=int, choices=[0,1,2], default=0, help='choose gpu')
parser.add_argument("--batch_size",type=int, default=2, help='batch size for dataloaders')

args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dir = f"ADSAM_{args.dataset_name}_{args.max_samples}_{args.loss}_crf_{args.apply_crf}_{timestamp}"
device = torch.device(f"cuda:{args.gpu}")

# Set up logging to file
log_filename = f"training_log_{dir}.txt"
sys.stdout = TeeLogger(log_filename)
print(f"Logging output to: {log_filename}")


def train_dual_encoder_cityscapes(
    sam_model,
    train_loader,
    val_loader,
    num_epochs=args.num_epochs,
    learning_rate=2e-4,
    device=device,
    save_path= f"dual_encoder_{dir}.pth",
    apply_crf = args.apply_crf
):
    # Initialize dual encoder model
    model = DualEncoderDeformableDecoder(sam_model, num_classes=19).to(device)
    model = model.float()
    
    # Freeze SAM encoder
    for param in model.sam_encoder.parameters():
        param.requires_grad = False
    
    # maybe freeze some resnet layers? 
    for param in model.resnet_encoder.parameters():
        param.requires_grad = True # Unfreeze all layers
        
    # Collect decoder params (upsample blocks + classifier)
    decoder_params = (
        list(model.up1.parameters()) +
        list(model.up2.parameters()) +
        list(model.up3.parameters()) +
        list(model.up4.parameters()) +
        list(model.classifier.parameters())
    )

    # Initialize optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': decoder_params, 'lr': learning_rate},
        {'params': model.fusion_layers.parameters(), 'lr': learning_rate},
        {'params': model.resnet_encoder.parameters(), 'lr': learning_rate * 0.1}
    ], weight_decay=0.0005)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Loss function
    
    if args.loss == "hybrid":
        criterion =  HybridLoss(num_classes=19)
    elif args.loss == "focaldice":
        criterion = FocalDiceLoss(
            alpha=0.25,
            gamma=2.0,
            dice_weight=0.5,
            ignore_index=19,
            num_classes=19
        )
    
    # Metrics
    train_metric = IoUMetric(num_classes=19)
    val_metric = IoUMetric(num_classes=19)
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler("cuda")
    
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (batch_input, target_masks) in enumerate(pbar):
            # Move data to device
            images = batch_input["image"].to(device, dtype=torch.float32)
            target_masks = target_masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast("cuda"):
                logits = model(images)
                
                # Resize logits if needed
                if logits.shape[-2:] != target_masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=target_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                loss = criterion(logits, target_masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item()
            pred_masks = torch.argmax(logits, dim=1)
            train_metric.update(pred_masks, target_masks)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_input, target_masks in tqdm(val_loader, desc='Validation'):
                images      = batch_input["image"].to(device)
                target_masks = target_masks.to(device)

                logits = model(images)
                # Resize if necessary
                if logits.shape[-2:] != target_masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=target_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Loss
                loss = criterion(logits, target_masks)
                val_loss += loss.item()

                # Convert to probabilities
                probs = torch.softmax(logits, dim=1)

    
                if apply_crf:
                    refined_masks = []
                    for img_tensor, prob_tensor in zip(images, probs):
                        # 1) grab the numpy image & prob map
                        img_np  = img_tensor.detach().cpu().numpy().transpose(1,2,0)
                        prob_np = prob_tensor.detach().cpu().numpy().transpose(1,2,0)
                        # 2) run CRF → this now returns a H×W label mask
                        refined_mask = crf_postprocess(img_np, prob_np, n_classes=probs.size(1), iter_max=10)
                        # 3) turn into tensor [1, H, W]
                        refined_tensor = torch.from_numpy(refined_mask).to(device).unsqueeze(0)
                        refined_masks.append(refined_tensor)
                    # 4) concat into [B, H, W] and update metric directly
                    pred_masks = torch.cat(refined_masks, dim=0)
                    val_metric.update(pred_masks, target_masks)
                else:
                    pred_masks = torch.argmax(probs, dim=1)
                    val_metric.update(pred_masks, target_masks)
        
        # Calculate metrics
        train_miou, train_class_ious = train_metric.compute()
        val_miou, val_class_ious = val_metric.compute()
        
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train mIOU: {train_miou:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val mIOU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, save_path)
            print(f"Saved new best model with mIOU: {best_miou:.4f}")
        
        # Visualize predictions
        if (epoch + 1) % 1 == 0:
            print("\nGenerating validation sample visualizations...")
            visualize_predictions_dual(
                model=model,
                val_loader=val_loader,
                epoch=epoch + 1,
                save_dir= f"run_lim_hybrid{dir}",
                num_samples=4,
                apply_crf=args.apply_crf
            )
        
        # Update learning rate
        scheduler.step()
        
        # Print per-class IoU
        for class_idx, iou in enumerate(val_class_ious):
            print(f"Class {class_idx} IoU: {iou:.4f}")
    
    return model


def load_sam_checkpoint(model: nn.Module, checkpoint_path: str):
    state_dict = torch.load(checkpoint_path, map_location=device)

    compatible_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }

    model.load_state_dict(compatible_state_dict, strict=False)

    for name, param in model.named_parameters():
        if name not in compatible_state_dict:
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    print("checkpoint loaded")
    return model



if __name__ == "__main__":
    sam_model = initialize_sam_model()
    if args.dataset_name.lower() == "cityscapes":
        train_images = CITYSCAPES_TRAIN_IMAGES
        train_masks = CITYSCAPES_TRAIN_MASKS
        val_images = CITYSCAPES_VAL_IMAGES
        val_masks = CITYSCAPES_VAL_MASKS
    elif args.dataset_name.lower() == "bdd100k":
        train_images = BDD100K_TRAIN_IMAGES
        train_masks = BDD100K_TRAIN_MASKS
        val_images = BDD100K_VAL_IMAGES
        val_masks = BDD100K_VAL_MASKS
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    
    # train_loader = get_cityscapes_dataloader(CITYSCAPES_TRAIN_IMAGES, CITYSCAPES_TRAIN_MASKS,max_samples=args.max_samples)
    # val_loader = get_cityscapes_dataloader(CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_MASKS,max_samples=(args.max_samples//2 if args.max_samples is not None else None))
    
    
    train_loader = get_dataloader(
        dataset_name=args.dataset_name,
        image_dir=train_images,
        mask_dir=train_masks,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    val_loader = get_dataloader(
        dataset_name=args.dataset_name,
        image_dir=val_images,
        mask_dir=val_masks,
        max_samples=(args.max_samples//2 if args.max_samples is not None else None),
        batch_size=args.batch_size,
    )    
    
    print(f"\n===+++===+++===+++===")
    print('# Dataset: {}'.format(args.dataset_name))
    print('# Dataset Size: {}'.format(args.max_samples))
    print('# Epochs: {}'.format(args.num_epochs))
    print(f"# Apply CRF: {args.apply_crf}")
    print(f"# Loss Function: {args.loss}")
    print(f"# GPU: {args.gpu}")
    print(f"===+++===+++===+++===")
    print('')
    
    model = MulticlassSam(sam_model=sam_model, num_classes=19)
    model = load_sam_checkpoint(model, SAM_CHECKPOINT_PATH)
    
    trained_model = train_dual_encoder_cityscapes(
        sam_model=sam_model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Close log file
    print(f"\nTraining complete. Log saved to: {log_filename}")
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()