import os
import random

import numpy as np
import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

from .cityscapes_dataset import IGNORE_INDEX, prepare_image

BDD_IGNORE_RAW = 255  # BDD100K train_id masks use 255 for ignore


def find_bdd100k_pairs(image_dir, annot_dir, max_samples=None):
    masks = sorted(
        os.path.join(annot_dir, f) for f in os.listdir(annot_dir) if f.endswith('_train_id.png')
    )
    if max_samples is not None:
        masks = masks[:max_samples]
    images = [
        os.path.join(image_dir, os.path.basename(f).replace('_train_id.png', '.jpg'))
        for f in masks
    ]
    return images, masks


def map_bdd_mask(mask_pil, do_flip, ignore_index=IGNORE_INDEX):
    arr = np.array(mask_pil).astype(np.int64)
    if do_flip:
        arr = arr[:, ::-1]
    arr = np.ascontiguousarray(arr)
    arr[arr == BDD_IGNORE_RAW] = ignore_index
    return torch.from_numpy(arr)


class BDD100kDataset(Dataset):
    """
    BDD100K (10K subset) with pre-rasterised train_id PNG masks (0-18 classes, 255 ignore).
    Returns ({"image", "original_size"[, "mask_native"]}, mask).
    """
    def __init__(self, image_dir, annot_dir, target_size=(1024, 1024), max_samples=None,
                 flip_augment=False, native_mask=False):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.target_size = target_size
        self.flip_augment = flip_augment
        self.native_mask = native_mask
        self.ignore_index = IGNORE_INDEX
        self.sam_transform = ResizeLongestSide(target_size[0])

        self.images, self.masks = find_bdd100k_pairs(image_dir, annot_dir, max_samples)
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask_pil = Image.open(self.masks[idx])
        orig_size = img.size

        do_flip = self.flip_augment and random.random() > 0.5
        img_t = prepare_image(img, self.target_size, self.sam_transform, do_flip)
        mask_t = map_bdd_mask(mask_pil.resize(self.target_size, Image.NEAREST), do_flip)

        batched_input = {"image": img_t, "original_size": orig_size}
        if self.native_mask:
            batched_input["mask_native"] = map_bdd_mask(mask_pil, do_flip)
        return batched_input, mask_t
