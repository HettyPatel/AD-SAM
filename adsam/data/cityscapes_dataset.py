import os
import random

import numpy as np
import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

IGNORE_INDEX = 19

# Cityscapes labelId -> train id (19 classes). Anything not listed -> ignore.
CITYSCAPES_LABEL_MAPPING = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_label_lut(mapping, ignore_index=IGNORE_INDEX, size=256):
    """uint8 label id -> train id lookup table (vectorised replacement for a dict loop)."""
    lut = np.full(size, ignore_index, dtype=np.int64)
    for k, v in mapping.items():
        lut[k] = v
    return lut


def find_cityscapes_pairs(image_dir, mask_dir, max_samples=None):
    images = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(image_dir)
        for name in files if name.endswith('leftImg8bit.png')
    )
    masks = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(mask_dir)
        for name in files if name.endswith('gtFine_labelIds.png')
    )
    if max_samples is not None:
        images, masks = images[:max_samples], masks[:max_samples]
    for img_path, mask_path in zip(images, masks):
        a = os.path.basename(img_path).replace('_leftImg8bit.png', '')
        b = os.path.basename(mask_path).replace('_gtFine_labelIds.png', '')
        assert a == b, f"Mismatch between {img_path} and {mask_path}"
    return images, masks


def prepare_image(image_pil, target_size, sam_transform, do_flip):
    """Resize, optional flip, SAM transform, ImageNet normalisation (ResNet branch)."""
    image_np = np.array(image_pil.resize(target_size, Image.BILINEAR))
    if do_flip:
        image_np = image_np[:, ::-1, :].copy()
    t = torch.from_numpy(sam_transform.apply_image(image_np)).permute(2, 0, 1).float() / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


class CityscapesDataset(Dataset):
    """Cityscapes for AD-SAM. Returns ({"image", "original_size"[, "mask_native"]}, mask)."""

    def __init__(self, image_dir, mask_dir, transform=None, target_size=(1024, 1024),
                 max_samples=None, flip_augment=False, native_mask=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.transform = transform
        self.flip_augment = flip_augment
        self.native_mask = native_mask
        self.ignore_index = IGNORE_INDEX
        self.label_mapping = CITYSCAPES_LABEL_MAPPING
        self._lut = build_label_lut(self.label_mapping, self.ignore_index)
        self.sam_transform = ResizeLongestSide(target_size[0])

        self.images, self.masks = find_cityscapes_pairs(image_dir, mask_dir, max_samples)
        print(f'Found {len(self.images)} images and {len(self.masks)} masks')

    def __len__(self):
        return len(self.images)

    def _map_mask(self, mask_pil, do_flip):
        arr = np.array(mask_pil, dtype=np.uint8)
        if do_flip:
            arr = arr[:, ::-1]
        return torch.from_numpy(self._lut[arr])  # fancy indexing yields a fresh contiguous array

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask_pil = Image.open(self.masks[idx])
        original_size = image.size

        do_flip = self.flip_augment and random.random() > 0.5
        input_image = prepare_image(image, self.target_size, self.sam_transform, do_flip)
        mask = self._map_mask(mask_pil.resize(self.target_size, Image.NEAREST), do_flip)

        batched_input = {"image": input_image, "original_size": original_size}
        if self.native_mask:
            batched_input["mask_native"] = self._map_mask(mask_pil, do_flip)
        return batched_input, mask
