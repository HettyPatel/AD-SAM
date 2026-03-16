"""
Pre-generate SAM image encoder embeddings (original + horizontally flipped).
Saves per-image .pt files so training skips the frozen encoder entirely.

Usage:
    python pregenerate_embeddings.py --dataset_name cityscapes --gpu 0
    python pregenerate_embeddings.py --dataset_name bdd100k --gpu 0
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm

from configs.paths import (
    CITYSCAPES_TRAIN_IMAGES, CITYSCAPES_TRAIN_MASKS,
    CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_MASKS,
    BDD100K_TRAIN_IMAGES, BDD100K_TRAIN_MASKS,
    BDD100K_VAL_IMAGES, BDD100K_VAL_MASKS,
    SAM_CHECKPOINT_PATH,
)
from utils.setup import initialize_sam_model


def prepare_image(image_path, sam_transform):
    """Load and preprocess a single image for SAM encoder (SAM normalization)."""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((1024, 1024), Image.BILINEAR)
    img_np = np.array(img_resized)
    sam_img = sam_transform.apply_image(img_np)
    # SAM normalization: 0-255 scale with SAM-specific mean/std
    img_t = torch.from_numpy(sam_img).permute(2, 0, 1).float()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    return img_t


def load_sam_encoder(device):
    """Load SAM model with checkpoint and return the image encoder."""
    sam_model = initialize_sam_model()
    state_dict = torch.load(SAM_CHECKPOINT_PATH, map_location='cpu')
    compatible = {
        k: v for k, v in state_dict.items()
        if k in sam_model.state_dict() and sam_model.state_dict()[k].shape == v.shape
    }
    sam_model.load_state_dict(compatible, strict=False)
    encoder = sam_model.image_encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def get_image_paths(dataset_name, split):
    """Return sorted list of image paths for a dataset split."""
    if dataset_name == 'cityscapes':
        if split == 'train':
            img_dir = CITYSCAPES_TRAIN_IMAGES
        else:
            img_dir = CITYSCAPES_VAL_IMAGES
        return sorted([
            os.path.join(root, name)
            for root, _, files in os.walk(img_dir)
            for name in files
            if name.endswith('leftImg8bit.png')
        ])
    elif dataset_name == 'bdd100k':
        if split == 'train':
            img_dir = BDD100K_TRAIN_IMAGES
            mask_dir = BDD100K_TRAIN_MASKS
        else:
            img_dir = BDD100K_VAL_IMAGES
            mask_dir = BDD100K_VAL_MASKS
        # Derive from mask files to stay consistent with BDD100kDataset
        masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith('_train_id.png')
        ])
        return [
            os.path.join(img_dir, f.replace('_train_id.png', '.jpg'))
            for f in masks
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def generate_embeddings(encoder, image_paths, output_dir, device):
    """Generate and save SAM embeddings for original and flipped images."""
    os.makedirs(output_dir, exist_ok=True)
    sam_transform = ResizeLongestSide(1024)

    for img_path in tqdm(image_paths, desc=f"Generating -> {output_dir}"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        orig_path = os.path.join(output_dir, f"{basename}.pt")
        flip_path = os.path.join(output_dir, f"{basename}_flip.pt")

        # Skip if both already exist
        if os.path.exists(orig_path) and os.path.exists(flip_path):
            continue

        img_t = prepare_image(img_path, sam_transform)  # [3, 1024, 1024]

        # Original
        if not os.path.exists(orig_path):
            with torch.no_grad(), torch.amp.autocast('cuda'):
                emb = encoder(img_t.unsqueeze(0).to(device))  # [1, 256, 64, 64]
            torch.save(emb.squeeze(0).cpu(), orig_path)

        # Horizontally flipped
        if not os.path.exists(flip_path):
            img_flip = img_t.flip(-1)  # flip width dimension
            with torch.no_grad(), torch.amp.autocast('cuda'):
                emb_flip = encoder(img_flip.unsqueeze(0).to(device))
            torch.save(emb_flip.squeeze(0).cpu(), flip_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['cityscapes', 'bdd100k'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_root', type=str,
                        default='embeddings',
                        help='Root directory for cached embeddings')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    print(f"Loading SAM encoder...")
    encoder = load_sam_encoder(device)

    total_start = time.time()

    for split in ['train', 'val']:
        image_paths = get_image_paths(args.dataset_name, split)
        output_dir = os.path.join(args.output_root, args.dataset_name, split)
        print(f"\n{args.dataset_name}/{split}: {len(image_paths)} images")

        split_start = time.time()
        generate_embeddings(encoder, image_paths, output_dir, device)
        split_elapsed = time.time() - split_start
        print(f"  {split} done in {split_elapsed:.1f}s "
              f"({split_elapsed/max(len(image_paths),1):.2f}s/image)")

    total_elapsed = time.time() - total_start
    print(f"\nTotal embedding generation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # Save timing report
    os.makedirs('results', exist_ok=True)
    timing_path = f"results/embedding_generation_time_{args.dataset_name}.txt"
    with open(timing_path, 'w') as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n")
        f.write(f"Includes original + horizontally flipped variants\n")
    print(f"Timing saved to {timing_path}")


if __name__ == '__main__':
    main()
