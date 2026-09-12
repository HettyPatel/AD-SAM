"""Shared pieces of the two training entry points: argument defaults, run bookkeeping, epoch loop helpers."""
import argparse
import csv
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

from adsam.constants import CLASS_NAMES
from adsam.utils.train_utils import TeeLogger, gpu_name, make_tb_writer


def none_or_int(x):
    return None if str(x).lower() == 'none' else int(x)


def add_common_args(parser):
    parser.add_argument('--dataset_name', type=str, choices=['cityscapes', 'bdd100k'], default='cityscapes')
    parser.add_argument('--max_samples', type=none_or_int, default=None, help='training images to use ("None" = all)')
    parser.add_argument('--max_val_samples', type=none_or_int, default=None, help='validation images to use (smoke tests)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--flip_augment', action='store_true', help='50%% random horizontal flip')
    parser.add_argument('--eval_native', action='store_true',
                        help='score validation at the native label resolution (logits upsampled); default 1024x1024')
    parser.add_argument('--output_dir', type=str, default='results', help='where logs, curves, checkpoints and TensorBoard files go')
    parser.add_argument('--seed', type=int, default=None, help='torch/numpy seed (default: unseeded)')
    return parser


class Run:
    """Creates the output layout for one training run and opens its log, curves CSV and TensorBoard writer."""

    def __init__(self, run_id, args, device):
        self.run_id = run_id
        self.args = args
        self.device = device
        self.eval_res = 'native' if args.eval_native else '1024x1024'
        out = args.output_dir
        os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(out, 'checkpoints'), exist_ok=True)
        self.log_path = os.path.join(out, 'logs', f'training_log_{run_id}.txt')
        self.curves_path = os.path.join(out, f'curves_{run_id}.csv')
        self.summary_path = os.path.join(out, 'results_summary.csv')
        self.vis_dir = os.path.join(out, f'vis_{run_id}')
        sys.stdout = TeeLogger(self.log_path)
        print(f"Logging output to: {self.log_path}")
        self.curve_fields = ['epoch', 'train_loss', 'train_miou', 'val_loss', 'val_miou', 'epoch_time_s'] + \
                            [f'{cn}_iou' for cn in CLASS_NAMES]
        self._curves_file = open(self.curves_path, 'w', newline='')
        self.curves = csv.DictWriter(self._curves_file, fieldnames=self.curve_fields)
        self.curves.writeheader()
        self.tb = make_tb_writer(run_id, root=os.path.join(out, 'tensorboard'))
        self.tb.add_text('config', ' | '.join(f'{k}={v}' for k, v in sorted(vars(args).items())), 0)

    def checkpoint_path(self, prefix):
        return os.path.join(self.args.output_dir, 'checkpoints', f'{prefix}_{self.run_id}.pth')

    def write_epoch(self, epoch, train_loss, train_miou, val_loss, val_miou, class_ious, epoch_time):
        row = {'epoch': epoch, 'train_loss': f"{train_loss:.4f}", 'train_miou': f"{train_miou:.4f}",
               'val_loss': f"{val_loss:.4f}", 'val_miou': f"{val_miou:.4f}", 'epoch_time_s': f"{epoch_time:.1f}"}
        for i, cn in enumerate(CLASS_NAMES):
            row[f'{cn}_iou'] = f"{class_ious[i]:.4f}"
        self.curves.writerow(row)
        self._curves_file.flush()

    def close(self):
        self._curves_file.close()
        self.tb.close()
        print(f"\nTraining complete. Log saved to: {self.log_path}")
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()


SUMMARY_FIELDS = [
    'run_id', 'model', 'dataset', 'train_samples', 'ablation', 'loss', 'crf',
    'batch_size', 'num_epochs', 'flip_augment', 'eval_res', 'seed',
    'best_val_miou', 'best_epoch', 'last10_mean_val_miou', 'final_val_miou',
    'total_train_time_min', 'gpu',
] + [f'{cn}_iou' for cn in CLASS_NAMES]


def summary_row(run, model_name, ablation, loss_name, crf, best_miou, best_epoch, last10, final, total_time, class_ious):
    a = run.args
    row = {
        'run_id': run.run_id, 'model': model_name, 'dataset': a.dataset_name,
        'train_samples': a.max_samples if a.max_samples is not None else 'full',
        'ablation': ablation, 'loss': loss_name, 'crf': str(crf),
        'batch_size': a.batch_size, 'num_epochs': a.num_epochs, 'flip_augment': str(a.flip_augment),
        'eval_res': run.eval_res, 'seed': '' if a.seed is None else a.seed,
        'best_val_miou': f"{best_miou:.4f}", 'best_epoch': best_epoch,
        'last10_mean_val_miou': f"{last10:.4f}", 'final_val_miou': f"{final:.4f}",
        'total_train_time_min': f"{total_time/60:.1f}", 'gpu': gpu_name(run.device),
    }
    for i, cn in enumerate(CLASS_NAMES):
        row[f'{cn}_iou'] = f"{class_ious[i]:.4f}"
    return row


def seed_everything(seed):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def score_batch(metric, logits, target_1024, batch_input, eval_native, device, pred_masks=None):
    """Update the IoU metric either at 1024x1024 or at the native label resolution."""
    if eval_native:
        target = batch_input["mask_native"].to(device)
        if pred_masks is None:
            pred = torch.argmax(F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False), dim=1)
        else:  # e.g. CRF output already at 1024x1024
            pred = F.interpolate(pred_masks.unsqueeze(1).float(), size=target.shape[-2:], mode='nearest').squeeze(1).long()
        metric.update(pred, target)
    else:
        pred = torch.argmax(logits, dim=1) if pred_masks is None else pred_masks
        metric.update(pred, target_1024)


def print_header(title, run, extra):
    print("\n===+++===+++===+++===")
    print(f"# {title}")
    print(f"# Run ID: {run.run_id}")
    a = run.args
    print(f"# Dataset: {a.dataset_name}   train samples: {a.max_samples if a.max_samples is not None else 'all'}")
    print(f"# Epochs: {a.num_epochs}   batch: {a.batch_size}   flip: {a.flip_augment}   eval: {run.eval_res}   seed: {a.seed}")
    print(f"# GPU: {a.gpu} ({gpu_name(run.device)})   torch {torch.__version__}")
    for line in extra:
        print(f"# {line}")
    print("===+++===+++===+++===\n")
