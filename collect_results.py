"""
Parse all training log files and collect results into CSVs.

Produces:
  results/results_summary.csv      - Best mIoU + per-class IoUs for every experiment
  results/training_curves.csv      - Per-epoch train/val loss and mIoU for every experiment
  results/efficiency/efficiency_results.csv  - (produced by compute_efficiency.py)

Usage:
    python collect_results.py
"""

import os
import re
import csv
import glob
from datetime import datetime

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]


def parse_log(log_path):
    """Parse a training log file and extract key metrics."""
    with open(log_path, 'r') as f:
        text = f.read()

    result = {
        'log_file': os.path.basename(log_path),
        'model': '',
        'dataset': '',
        'train_samples': '',
        'ablation': 'full',
        'loss': '',
        'crf': 'False',
        'batch_size': '',
        'best_val_miou': 0.0,
        'best_epoch': 0,
        'final_train_miou': 0.0,
        'final_val_miou': 0.0,
        'total_epochs': 0,
    }
    for cn in CLASS_NAMES:
        result[f'{cn}_iou'] = 0.0

    # Detect model type
    if 'DeepLabV3' in log_path or 'DeepLabV3' in text:
        result['model'] = 'DeepLabV3-ResNet101'
        result['loss'] = 'ce'
    else:
        result['model'] = 'AD-SAM'

    # Parse header info
    m = re.search(r'# Dataset:\s*(\S+)', text)
    if m:
        result['dataset'] = m.group(1)

    m = re.search(r'# Dataset Size:\s*(\S+)', text)
    if m:
        val = m.group(1)
        result['train_samples'] = val if val != 'None' else 'full'

    m = re.search(r'# Loss Function:\s*(\S+)', text)
    if m:
        result['loss'] = m.group(1)

    m = re.search(r'# Ablation:\s*(\S+)', text)
    if m:
        result['ablation'] = m.group(1)

    m = re.search(r'# Apply CRF:\s*(\S+)', text)
    if m:
        result['crf'] = m.group(1)

    m = re.search(r'# Batch Size:\s*(\S+)', text)
    if m:
        result['batch_size'] = m.group(1)

    # Parse ALL epoch results
    # Pattern: "Epoch N" followed by train/val lines, then optional save line, then per-class
    epoch_pattern = re.compile(
        r'Epoch\s+(\d+)\n'
        r'Train Loss:\s*([\d.]+),\s*Train mIOU:\s*([\d.]+)\n'
        r'Val Loss:\s*([\d.]+),\s*Val mIOU:\s*([\d.]+)\n'
        r'(?:(?!Epoch\s+\d|Class \d).*\n)*?'  # skip any lines between val and class IoUs
        r'((?:Class \d+ IoU:.*?\n)*)',
        re.MULTILINE
    )

    best_val_miou = 0.0
    best_epoch = 0
    best_class_ious = [0.0] * 19
    final_train_miou = 0.0
    final_val_miou = 0.0
    epoch_data = []  # for training curves

    for match in epoch_pattern.finditer(text):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_miou = float(match.group(3))
        val_loss = float(match.group(4))
        val_miou = float(match.group(5))
        class_block = match.group(6)

        final_train_miou = train_miou
        final_val_miou = val_miou

        # Parse per-class IoUs for this epoch
        class_ious = [0.0] * 19
        for cidx, ciou in re.findall(r'Class (\d+) IoU:\s*([\d.]+)', class_block):
            idx = int(cidx)
            if idx < 19:
                class_ious[idx] = float(ciou)

        epoch_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_miou': train_miou,
            'val_loss': val_loss,
            'val_miou': val_miou,
        })

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            best_class_ious = class_ious

    result['best_val_miou'] = f"{best_val_miou:.4f}"
    result['best_epoch'] = best_epoch
    result['final_train_miou'] = f"{final_train_miou:.4f}"
    result['final_val_miou'] = f"{final_val_miou:.4f}"
    result['total_epochs'] = len(epoch_data)

    for i, cn in enumerate(CLASS_NAMES):
        result[f'{cn}_iou'] = f"{best_class_ious[i]:.4f}"

    return result, epoch_data


def make_experiment_id(result):
    """Short unique ID for an experiment row."""
    parts = [result['model'], result['dataset'], str(result['train_samples'])]
    if result['ablation'] != 'full':
        parts.append(result['ablation'])
    if result['crf'] == 'True':
        parts.append('crf')
    return '_'.join(parts)


def main():
    # Find all training log files (in current dir and results/logs/)
    log_files = sorted(
        glob.glob("training_log_*.txt") +
        glob.glob("results/logs/training_log_*.txt")
    )
    # Deduplicate by basename
    seen = set()
    unique_logs = []
    for lf in log_files:
        bn = os.path.basename(lf)
        if bn not in seen:
            seen.add(bn)
            unique_logs.append(lf)
    log_files = unique_logs

    if not log_files:
        print("No training log files found.")
        return

    print(f"Found {len(log_files)} log files")

    results = []
    all_curves = []

    for lf in log_files:
        try:
            r, epoch_data = parse_log(lf)
            results.append(r)

            exp_id = make_experiment_id(r)
            for ed in epoch_data:
                ed['experiment'] = exp_id
                ed['model'] = r['model']
                ed['dataset'] = r['dataset']
                ed['train_samples'] = r['train_samples']
                ed['ablation'] = r['ablation']
                all_curves.append(ed)

            print(f"  Parsed: {os.path.basename(lf)} -> {r['model']} {r['dataset']} "
                  f"samples={r['train_samples']} ablation={r['ablation']} "
                  f"best_mIoU={r['best_val_miou']} ({r['total_epochs']} epochs)")
        except Exception as e:
            print(f"  ERROR parsing {lf}: {e}")

    if not results:
        print("No results to write.")
        return

    os.makedirs("results", exist_ok=True)

    # ── 1. Results summary CSV ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/results_summary_{timestamp}.csv"
    fieldnames = [
        'log_file', 'model', 'dataset', 'train_samples', 'ablation', 'loss',
        'crf', 'batch_size', 'best_val_miou', 'best_epoch', 'final_train_miou',
        'final_val_miou', 'total_epochs'
    ] + [f'{cn}_iou' for cn in CLASS_NAMES]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults summary written to {csv_path}")

    # ── 2. Training curves CSV ───────────────────────────────────────────────
    curves_path = f"results/training_curves_{timestamp}.csv"
    curve_fields = ['experiment', 'model', 'dataset', 'train_samples', 'ablation',
                    'epoch', 'train_loss', 'train_miou', 'val_loss', 'val_miou']
    with open(curves_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(all_curves)
    print(f"Training curves written to {curves_path} ({len(all_curves)} rows)")

    # ── 3. Print summary table ───────────────────────────────────────────────
    print(f"\n{'Model':<25} {'Dataset':<12} {'Samples':<8} {'Ablation':<14} "
          f"{'Loss':<8} {'CRF':<6} {'Best mIoU':<10} {'Epoch':<6} {'Total':<6}")
    print("-" * 105)
    for r in sorted(results, key=lambda x: (x['model'], x['dataset'], x['train_samples'])):
        print(f"{r['model']:<25} {r['dataset']:<12} {str(r['train_samples']):<8} "
              f"{r['ablation']:<14} {r['loss']:<8} {r['crf']:<6} "
              f"{r['best_val_miou']:<10} {str(r['best_epoch']):<6} {str(r['total_epochs']):<6}")

    # ── 4. Check if efficiency results exist ─────────────────────────────────
    eff_path = "results/efficiency/efficiency_results.csv"
    if os.path.exists(eff_path):
        print(f"\nEfficiency results: {eff_path}")
    else:
        print(f"\nNote: No efficiency results found at {eff_path} — run compute_efficiency.py")


if __name__ == "__main__":
    main()
