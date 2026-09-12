"""DeepLabV3-ResNet101 baseline. Run as:  python train.py deeplabv3 [options]
Same datasets, validation set, metric and output format as AD-SAM."""
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from adsam.config import dataset_paths
from adsam.constants import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from adsam.data.dataloaders import get_dataloader
from adsam.metrics.iou_metric import IoUMetric
from adsam.utils.train_utils import freeze_bn, save_checkpoint, last_n_mean, append_summary_row, tb_log_epoch
from adsam.training.common import (add_common_args, Run, SUMMARY_FIELDS, summary_row, seed_everything,
                                   score_batch, print_header)


def build_parser():
    p = argparse.ArgumentParser(prog='train.py deeplabv3', description='Train the DeepLabV3-ResNet101 baseline.')
    add_common_args(p)
    p.add_argument('--lr', type=float, default=0.01, help='head learning rate (backbone gets 0.1x, BatchNorm stats frozen)')
    return p


def build_deeplabv3(num_classes=NUM_CLASSES):
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, 1)
    return model


def poly_lr(optimizer, it, max_it, power=0.9):
    factor = (1 - it / max_it) ** power
    for g in optimizer.param_groups:
        g['lr'] = g['base_lr'] * factor
    return optimizer.param_groups[0]['lr']


def train(args, run, model, train_loader, val_loader):
    device = run.device
    model = model.to(device).float()
    head_lr, backbone_lr = args.lr, args.lr * 0.1
    head_params = list(model.classifier.parameters()) + list(model.aux_classifier.parameters())
    optimizer = torch.optim.SGD([
        {'params': model.backbone.parameters(), 'lr': backbone_lr, 'base_lr': backbone_lr},
        {'params': head_params, 'lr': head_lr, 'base_lr': head_lr},
    ], momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = GradScaler('cuda')
    train_metric, val_metric = IoUMetric(NUM_CLASSES), IoUMetric(NUM_CLASSES)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M trainable")
    print(f"LR: heads {head_lr}, backbone {backbone_lr} (0.1x), poly decay; backbone BatchNorm statistics frozen")

    best_miou, best_epoch, best_class_ious, val_history = 0.0, 0, [0.0] * NUM_CLASSES, []
    save_path = run.checkpoint_path('deeplabv3')
    total_iters, it, global_step = args.num_epochs * len(train_loader), 0, 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        t_epoch = time.time()
        model.train()
        freeze_bn(model.backbone)
        train_metric.reset()
        train_loss = 0.0
        for batch_input, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images = batch_input['image'].to(device, dtype=torch.float32)
            target = target.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(images)
                logits = F.interpolate(out['out'], size=target.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(logits, target)
                if 'aux' in out:
                    aux = F.interpolate(out['aux'], size=target.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss + 0.4 * criterion(aux, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss ({loss.item()}) at epoch {epoch+1}, step {global_step+1}; aborting")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            it += 1
            poly_lr(optimizer, it, total_iters)
            train_loss += loss.item()
            global_step += 1
            if global_step % 10 == 0:
                run.tb.add_scalar('loss/train_step', loss.item(), global_step)
            train_metric.update(torch.argmax(logits, dim=1), target)

        model.eval()
        val_metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for batch_input, target in tqdm(val_loader, desc='Validation'):
                images = batch_input['image'].to(device, dtype=torch.float32)
                target = target.to(device)
                logits = F.interpolate(model(images)['out'], size=target.shape[-2:], mode='bilinear', align_corners=False)
                val_loss += criterion(logits, target).item()
                score_batch(val_metric, logits, target, batch_input, args.eval_native, device)

        train_miou, _ = train_metric.compute()
        val_miou, val_class_ious = val_metric.compute()
        val_history.append(float(val_miou))
        epoch_time = time.time() - t_epoch
        tl, vl = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}\nTrain Loss: {tl:.4f}, Train mIoU: {train_miou:.4f}\nVal Loss: {vl:.4f}, Val mIoU: {val_miou:.4f}  (eval {run.eval_res})")
        print(f"Epoch time: {epoch_time/60:.1f} min  (elapsed {(time.time()-t_start)/60:.1f} min)")
        if val_miou > best_miou:
            best_miou, best_epoch, best_class_ious = val_miou, epoch + 1, list(val_class_ious)
            save_checkpoint(save_path, model, epoch, best_miou, extra={'run_id': run.run_id})
            print(f"Saved new best model with mIoU: {best_miou:.4f}")
        for i, iou in enumerate(val_class_ious):
            print(f"Class {i} ({CLASS_NAMES[i]}) IoU: {iou:.4f}")
        tb_log_epoch(run.tb, epoch + 1, tl, float(train_miou), vl, float(val_miou), val_class_ious, CLASS_NAMES,
                     optimizer.param_groups[0]['lr'], epoch_time)
        run.write_epoch(epoch + 1, tl, train_miou, vl, val_miou, val_class_ious, epoch_time)

    total_time = time.time() - t_start
    last10, final = last_n_mean(val_history, 10), (val_history[-1] if val_history else 0.0)
    row = summary_row(run, 'DeepLabV3-ResNet101', 'full', 'ce', False, best_miou, best_epoch, last10, final, total_time, best_class_ious)
    written = append_summary_row(run.summary_path, SUMMARY_FIELDS, row)
    print(f"\nBest val mIoU {best_miou:.4f} at epoch {best_epoch}; last-10 mean {last10:.4f}; final {final:.4f}")
    print(f"Total training time: {total_time/60:.1f} min")
    print(f"Results saved to {run.curves_path} and {written}")
    return model


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"DeepLabV3_{args.dataset_name}_{args.max_samples if args.max_samples is not None else 'full'}_{stamp}"
    run = Run(run_id, args, device)
    train_images, train_masks, val_images, val_masks = dataset_paths(args.dataset_name)
    train_loader = get_dataloader(args.dataset_name, train_images, train_masks, batch_size=args.batch_size,
                                  max_samples=args.max_samples, flip_augment=args.flip_augment, is_train=True)
    val_loader = get_dataloader(args.dataset_name, val_images, val_masks, batch_size=args.batch_size,
                                max_samples=args.max_val_samples, flip_augment=False, is_train=False,
                                native_mask=args.eval_native)
    print_header('DeepLabV3-ResNet101 baseline', run, ["torchvision COCO-pretrained weights, 19-class heads, cross-entropy + 0.4 aux"])
    train(args, run, build_deeplabv3(), train_loader, val_loader)
    run.close()


if __name__ == '__main__':
    main()
