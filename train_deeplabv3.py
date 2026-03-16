"""
DeepLabV3-ResNet101 baseline training script.

Uses the SAME datasets, val set, IoU metric, and logging format as AD-SAM
for fair comparison.

Usage:
    python train_deeplabv3.py --dataset_name cityscapes --num_epochs 100 --batch_size 4 --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision
import csv
import os
import sys
import argparse
from datetime import datetime

from data.dataloaders import get_dataloader
from metrics.iou_metric import IoUMetric
from configs.paths import (CITYSCAPES_TRAIN_IMAGES, CITYSCAPES_TRAIN_MASKS,
                           CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_MASKS,
                           BDD100K_TRAIN_IMAGES, BDD100K_TRAIN_MASKS,
                           BDD100K_VAL_IMAGES, BDD100K_VAL_MASKS)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole",
               "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
               "person", "rider", "car", "truck", "bus", "train_cls", "motorcycle", "bicycle"]


class TeeLogger:
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
        self._closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self._closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if not self._closed:
            self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        if not self._closed:
            self._closed = True
            self.log.close()


def build_deeplabv3(num_classes=19):
    """DeepLabV3-ResNet101 with pretrained backbone."""
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    )
    # Replace classifier heads for our class count
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)
    return model


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    """Polynomial learning rate decay (standard for DeepLabV3)."""
    lr = init_lr * ((1 - iter / max_iter) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_deeplabv3(model, train_loader, val_loader, args, device,
                    save_path, curves_csv_path, summary_csv_path, run_name):
    model = model.to(device).float()

    init_lr = 0.01
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss(ignore_index=19)
    scaler = GradScaler("cuda")

    train_metric = IoUMetric(num_classes=19)
    val_metric = IoUMetric(num_classes=19)

    best_miou = 0.0
    best_class_ious = [0.0] * 19
    total_iters = args.num_epochs * len(train_loader)
    current_iter = 0

    # Initialize per-epoch CSV
    curve_fields = ['epoch', 'train_loss', 'train_miou', 'val_loss', 'val_miou'] + [f'{cn}_iou' for cn in CLASS_NAMES]
    curves_csv = open(curves_csv_path, 'w', newline='')
    curves_writer = csv.DictWriter(curves_csv, fieldnames=curve_fields)
    curves_writer.writeheader()

    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_metric.reset()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch_idx, (batch_input, target_masks) in enumerate(pbar):
            images = batch_input["image"].to(device, dtype=torch.float32)
            target_masks = target_masks.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):
                output = model(images)
                logits = output['out']

                if logits.shape[-2:] != target_masks.shape[-2:]:
                    logits = F.interpolate(logits, size=target_masks.shape[-2:],
                                           mode='bilinear', align_corners=False)

                loss = criterion(logits, target_masks)

                # Aux loss
                if 'aux' in output:
                    aux_logits = output['aux']
                    if aux_logits.shape[-2:] != target_masks.shape[-2:]:
                        aux_logits = F.interpolate(aux_logits, size=target_masks.shape[-2:],
                                                   mode='bilinear', align_corners=False)
                    loss = loss + 0.4 * criterion(aux_logits, target_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_iter += 1
            current_lr = poly_lr_scheduler(optimizer, init_lr, current_iter, total_iters)

            train_loss += loss.item()
            pred_masks = torch.argmax(logits, dim=1)
            train_metric.update(pred_masks, target_masks)

            pbar.set_postfix({'loss': loss.item(), 'lr': current_lr})

        # Validation
        model.eval()
        val_metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch_input, target_masks in tqdm(val_loader, desc='Validation'):
                images = batch_input["image"].to(device, dtype=torch.float32)
                target_masks = target_masks.to(device)

                output = model(images)
                logits = output['out']

                if logits.shape[-2:] != target_masks.shape[-2:]:
                    logits = F.interpolate(logits, size=target_masks.shape[-2:],
                                           mode='bilinear', align_corners=False)

                loss = criterion(logits, target_masks)
                val_loss += loss.item()

                pred_masks = torch.argmax(logits, dim=1)
                val_metric.update(pred_masks, target_masks)

        train_miou, train_class_ious = train_metric.compute()
        val_miou, val_class_ious = val_metric.compute()

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train mIOU: {train_miou:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val mIOU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_class_ious = list(val_class_ious)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, save_path)
            print(f"Saved new best model with mIOU: {best_miou:.4f}")

        # Per-class IoU
        for class_idx, iou in enumerate(val_class_ious):
            print(f"Class {class_idx} IoU: {iou:.4f}")

        # Write epoch row to curves CSV
        row = {
            'epoch': epoch + 1,
            'train_loss': f"{train_loss/len(train_loader):.4f}",
            'train_miou': f"{train_miou:.4f}",
            'val_loss': f"{val_loss/len(val_loader):.4f}",
            'val_miou': f"{val_miou:.4f}",
        }
        for i, cn in enumerate(CLASS_NAMES):
            row[f'{cn}_iou'] = f"{val_class_ious[i]:.4f}"
        curves_writer.writerow(row)
        curves_csv.flush()

    curves_csv.close()

    # Append summary row to shared results CSV
    summary_exists = os.path.exists(summary_csv_path)
    summary_fields = [
        'run_id', 'model', 'dataset', 'train_samples', 'ablation', 'loss', 'crf',
        'batch_size', 'num_epochs', 'best_val_miou', 'best_epoch',
    ] + [f'{cn}_iou' for cn in CLASS_NAMES]
    with open(summary_csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        if not summary_exists:
            writer.writeheader()
        summary_row = {
            'run_id': run_name,
            'model': 'DeepLabV3-ResNet101',
            'dataset': args.dataset_name,
            'train_samples': args.max_samples if args.max_samples is not None else 'full',
            'ablation': 'full',
            'loss': 'ce',
            'crf': 'False',
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'best_val_miou': f"{best_miou:.4f}",
            'best_epoch': int(epoch + 1) if best_miou > 0 else 0,
        }
        for i, cn in enumerate(CLASS_NAMES):
            summary_row[f'{cn}_iou'] = f"{best_class_ious[i]:.4f}"
        writer.writerow(summary_row)

    print(f"\nResults saved to {curves_csv_path} and {summary_csv_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=lambda x: None if x.lower() == 'none' else int(x), default=None, help='max training samples (use "None" for full dataset)')
    parser.add_argument('--dataset_name', type=str, choices=['cityscapes', 'bdd100k'],
                        default='cityscapes')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--max_val_samples", type=lambda x: None if x.lower() == 'none' else int(x), default=None, help="limit val samples (default: None = full val set)")
    parser.add_argument("--flip_augment", action="store_true", default=False, help="enable horizontal flip augmentation")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DeepLabV3_{args.dataset_name}_{args.max_samples}_{timestamp}"
    device = torch.device(f"cuda:{args.gpu}")

    # Logging
    log_filename = f"training_log_{run_name}.txt"
    sys.stdout = TeeLogger(log_filename)
    print(f"Logging output to: {log_filename}")

    save_path = f"deeplabv3_{run_name}.pth"

    # CSV paths
    os.makedirs("results", exist_ok=True)
    curves_csv_path = f"results/curves_{run_name}.csv"
    summary_csv_path = "results/results_summary.csv"

    # Dataset paths
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

    train_loader = get_dataloader(
        dataset_name=args.dataset_name,
        image_dir=train_images,
        mask_dir=train_masks,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        flip_augment=args.flip_augment,
    )
    val_loader = get_dataloader(
        dataset_name=args.dataset_name,
        image_dir=val_images,
        mask_dir=val_masks,
        max_samples=args.max_val_samples,  # None = full val set; override for smoke tests
        batch_size=args.batch_size,
        flip_augment=False,  # no augmentation during validation
    )

    print(f"\n===+++===+++===+++===")
    print(f"# Model: DeepLabV3-ResNet101")
    print(f"# Dataset: {args.dataset_name}")
    print(f"# Dataset Size: {args.max_samples}")
    print(f"# Epochs: {args.num_epochs}")
    print(f"# Batch Size: {args.batch_size}")
    print(f"# GPU: {args.gpu}")
    print(f"===+++===+++===+++===\n")

    model = build_deeplabv3(num_classes=19)
    trained_model = train_deeplabv3(model, train_loader, val_loader, args, device,
                                    save_path, curves_csv_path, summary_csv_path, run_name)

    print(f"\nTraining complete. Log saved to: {log_filename}")
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
