"""Helpers shared by train.py and train_deeplabv3.py."""
import csv
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn


class TeeLogger:
    """Mirror stdout to a log file."""
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


def freeze_bn(module):
    """Put every BatchNorm layer under `module` in eval mode so running statistics
    are not updated. Call after model.train(). Affine weights stay trainable."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def gpu_name(device):
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        return 'cpu'


def lean_state_dict(model):
    """State dict without frozen parameters (e.g. the SAM encoder).

    Buffers of trainable submodules (BatchNorm running stats, normalisation
    constants) are kept; buffers of fully frozen submodules are dropped.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    frozen_roots = {n.split('.')[0] for n in frozen} - {n.split('.')[0] for n in trainable}
    out = {}
    for k, v in model.state_dict().items():
        if k in trainable:
            out[k] = v
        elif k in frozen or k.split('.')[0] in frozen_roots:
            continue
        else:
            out[k] = v
    return out


def save_checkpoint(path, model, epoch, best_miou, extra=None):
    """Save trainable weights only (no optimizer state)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'epoch': epoch, 'best_miou': best_miou, 'model_state_dict': lean_state_dict(model)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path, map_location='cpu'):
    """Load a lean checkpoint; frozen parts (SAM encoder) come from their own weights."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    result = model.load_state_dict(ck['model_state_dict'], strict=False)
    return ck, result


def last_n_mean(values, n=10):
    tail = list(values)[-n:]
    return sum(tail) / len(tail) if tail else 0.0


def append_summary_row(path, fields, row):
    """Append a row to the shared results CSV. If an existing file has a different
    header (older format), write to a timestamped sibling instead of corrupting it."""
    header = ','.join(fields)
    if os.path.exists(path):
        with open(path) as f:
            existing = f.readline().strip()
        if existing and existing != header:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = path.replace('.csv', f'_{stamp}.csv')
            print(f"results summary header changed; writing to {path}")
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


class _NoOpWriter:
    def add_scalar(self, *a, **k): pass
    def add_scalars(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def make_tb_writer(run_id, root='results/tensorboard'):
    """TensorBoard SummaryWriter under results/tensorboard/<run_id>; no-op if tensorboard is missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed; skipping TensorBoard logging")
        return _NoOpWriter()
    log_dir = os.path.join(root, run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs: {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def tb_log_epoch(writer, epoch, train_loss, train_miou, val_loss, val_miou, class_ious,
                 class_names, lr, epoch_time_s):
    """One call per epoch: scalars for the four curves, LR, timing and per-class val IoU."""
    writer.add_scalar('loss/train', train_loss, epoch)
    writer.add_scalar('loss/val', val_loss, epoch)
    writer.add_scalar('miou/train', train_miou, epoch)
    writer.add_scalar('miou/val', val_miou, epoch)
    writer.add_scalar('lr', lr, epoch)
    writer.add_scalar('time/epoch_s', epoch_time_s, epoch)
    for name, iou in zip(class_names, class_ious):
        writer.add_scalar(f'iou_val/{name}', float(iou), epoch)
    writer.flush()
