"""AD-SAM training. Run as:  python train.py adsam [options]"""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from adsam.ablations import build_ablation_model
from adsam.config import dataset_paths
from adsam.constants import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from adsam.data.dataloaders import get_dataloader
from adsam.losses.focal_dice import FocalDiceLoss
from adsam.losses.hybrid_loss import HybridLoss
from adsam.metrics.iou_metric import IoUMetric
from adsam.utils.crf_postprocess import crf_postprocess
from adsam.utils.setup import build_sam_for_training
from adsam.utils.train_utils import freeze_bn, save_checkpoint, last_n_mean, append_summary_row, tb_log_epoch
from adsam.utils.visualize import visualize_predictions_dual
from adsam.training.common import (add_common_args, Run, SUMMARY_FIELDS, summary_row, seed_everything,
                                   score_batch, print_header)


def build_parser():
    p = argparse.ArgumentParser(prog='train.py adsam', description='Train AD-SAM (frozen SAM ViT-H + ResNet-50 dual encoder).')
    add_common_args(p)
    p.add_argument('--embedding_dir', type=str, default=None,
                   help='root of pre-generated SAM embeddings (scripts/pregenerate_embeddings.py); omit to run the encoder live')
    p.add_argument('--ablation', type=str, default='full',
                   choices=['full', 'no_deform', 'no_attention', 'sam_encoder_only', 'ce_loss'])
    p.add_argument('--loss', type=str, choices=['hybrid', 'focaldice'], default='hybrid')
    p.add_argument('--lr', type=float, default=2e-4, help='learning rate for fusion + decoder (ResNet gets 0.1x)')
    p.add_argument('--apply_crf', action='store_true', help='DenseCRF post-processing on validation predictions')
    p.add_argument('--vis_every', type=int, default=25, help='save validation visualisations every N epochs (0 = off)')
    return p


def train(args, run, sam_model, train_loader, val_loader):
    device = run.device
    model = build_ablation_model(args.ablation, sam_model, num_classes=NUM_CLASSES).to(device).float()
    for p in model.sam_encoder.parameters():
        p.requires_grad = False

    if args.ablation == 'sam_encoder_only':
        optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr}],
                                      weight_decay=5e-4)
    else:
        decoder_params = [p for n, p in model.named_parameters() if n.startswith(('up1.', 'up2.', 'up3.', 'classifier.'))]
        optimizer = torch.optim.AdamW([
            {'params': decoder_params, 'lr': args.lr},
            {'params': model.fusion_layers.parameters(), 'lr': args.lr},
            {'params': model.resnet_encoder.parameters(), 'lr': args.lr * 0.1},
        ], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    if args.ablation == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.loss == 'hybrid':
        criterion = HybridLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    else:
        criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.5, ignore_index=IGNORE_INDEX, num_classes=NUM_CLASSES)

    train_metric, val_metric = IoUMetric(NUM_CLASSES), IoUMetric(NUM_CLASSES)
    scaler = GradScaler('cuda')
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_trainable/1e6:.2f}M trainable / {sum(p.numel() for p in model.parameters())/1e6:.2f}M total")

    best_miou, best_epoch, best_class_ious, val_history = 0.0, 0, [0.0] * NUM_CLASSES, []
    save_path = run.checkpoint_path('adsam')
    global_step = 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        t_epoch = time.time()
        model.train()
        if hasattr(model, 'resnet_encoder'):
            freeze_bn(model.resnet_encoder)   # small batches: keep the pretrained BatchNorm statistics
        train_metric.reset()
        train_loss = 0.0
        for batch_input, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images = batch_input['image'].to(device, dtype=torch.float32)
            target = target.to(device)
            sam_emb = batch_input.get('sam_embedding')
            if sam_emb is not None:
                sam_emb = sam_emb.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(images, sam_embedding=sam_emb)
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss ({loss.item()}) at epoch {epoch+1}, step {global_step+1}; aborting")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                sam_emb = batch_input.get('sam_embedding')
                if sam_emb is not None:
                    sam_emb = sam_emb.to(device, dtype=torch.float32)
                logits = model(images, sam_embedding=sam_emb)
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
                val_loss += criterion(logits, target).item()
                pred = None
                if args.apply_crf:
                    probs = torch.softmax(logits, dim=1)
                    refined = [torch.from_numpy(crf_postprocess(im.cpu().numpy().transpose(1, 2, 0), pr.cpu().numpy().transpose(1, 2, 0),
                                                                n_classes=NUM_CLASSES, iter_max=10)).to(device)
                               for im, pr in zip(images, probs)]
                    pred = torch.stack(refined, dim=0)
                score_batch(val_metric, logits, target, batch_input, args.eval_native, device, pred_masks=pred)

        train_miou, _ = train_metric.compute()
        val_miou, val_class_ious = val_metric.compute()
        val_history.append(float(val_miou))
        epoch_time = time.time() - t_epoch
        tl, vl = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}\nTrain Loss: {tl:.4f}, Train mIoU: {train_miou:.4f}\nVal Loss: {vl:.4f}, Val mIoU: {val_miou:.4f}  (eval {run.eval_res})")
        print(f"Epoch time: {epoch_time/60:.1f} min  (elapsed {(time.time()-t_start)/60:.1f} min)")

        if val_miou > best_miou:
            best_miou, best_epoch, best_class_ious = val_miou, epoch + 1, list(val_class_ious)
            save_checkpoint(save_path, model, epoch, best_miou, extra={'run_id': run.run_id, 'ablation': args.ablation})
            print(f"Saved new best model with mIoU: {best_miou:.4f}")
        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            visualize_predictions_dual(model=model, val_loader=val_loader, epoch=epoch + 1, save_dir=run.vis_dir,
                                       num_samples=4, apply_crf=args.apply_crf)
        scheduler.step()
        for i, iou in enumerate(val_class_ious):
            print(f"Class {i} ({CLASS_NAMES[i]}) IoU: {iou:.4f}")
        tb_log_epoch(run.tb, epoch + 1, tl, float(train_miou), vl, float(val_miou), val_class_ious, CLASS_NAMES,
                     optimizer.param_groups[0]['lr'], epoch_time)
        run.write_epoch(epoch + 1, tl, train_miou, vl, val_miou, val_class_ious, epoch_time)

    total_time = time.time() - t_start
    last10, final = last_n_mean(val_history, 10), (val_history[-1] if val_history else 0.0)
    row = summary_row(run, 'AD-SAM', args.ablation, 'ce' if args.ablation == 'ce_loss' else args.loss, args.apply_crf,
                      best_miou, best_epoch, last10, final, total_time, best_class_ious)
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
    run_id = f"ADSAM_{args.dataset_name}_{args.max_samples if args.max_samples is not None else 'full'}_{args.ablation}_{stamp}"
    run = Run(run_id, args, device)

    train_images, train_masks, val_images, val_masks = dataset_paths(args.dataset_name)
    train_emb = val_emb = None
    if args.embedding_dir:
        train_emb = os.path.join(args.embedding_dir, args.dataset_name, 'train')
        val_emb = os.path.join(args.embedding_dir, args.dataset_name, 'val')
    train_loader = get_dataloader(args.dataset_name, train_images, train_masks, batch_size=args.batch_size,
                                  max_samples=args.max_samples, embedding_dir=train_emb,
                                  flip_augment=args.flip_augment, is_train=True)
    val_loader = get_dataloader(args.dataset_name, val_images, val_masks, batch_size=args.batch_size,
                                max_samples=args.max_val_samples, embedding_dir=val_emb,
                                flip_augment=False, is_train=False, native_mask=args.eval_native)

    print_header('AD-SAM', run, [f"Ablation: {args.ablation}   loss: {'ce' if args.ablation == 'ce_loss' else args.loss}   CRF: {args.apply_crf}",
                                 f"Cached SAM embeddings: {args.embedding_dir or 'disabled (encoder runs live)'}",
                                 "ResNet-50 BatchNorm statistics frozen; SAM encoder frozen"])
    sam_model = build_sam_for_training(cached=bool(args.embedding_dir))
    train(args, run, sam_model, train_loader, val_loader)
    run.close()


if __name__ == '__main__':
    main()
