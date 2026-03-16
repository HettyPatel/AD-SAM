import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide

class BDD100kDataset(Dataset):
    """
    BDD100k Dataset for Semantic Segmentation using pre-rasterized PNG masks.
    Masks use Cityscapes train_id format (0-18 = classes, 255 = ignore).
    Images are JPGs from the 10K subset.
    """
    def __init__(
        self,
        image_dir: str,
        annot_dir: str,
        target_size=(1024, 1024),
        max_samples: int = None,
        flip_augment: bool = False,
    ):
        self.image_dir   = image_dir
        self.annot_dir   = annot_dir
        self.target_size = target_size
        self.flip_augment = flip_augment
        self.ignore_index = 19

        # SAM's resize transform
        self.sam_transform = ResizeLongestSide(target_size[0])

        # Gather mask files (*_train_id.png) and derive matching image paths
        self.masks = sorted([
            os.path.join(annot_dir, f)
            for f in os.listdir(annot_dir)
            if f.endswith('_train_id.png')
        ])
        if max_samples is not None:
            self.masks = self.masks[:max_samples]

        # Derive image paths: strip _train_id.png -> add .jpg
        self.images = [
            os.path.join(image_dir, os.path.basename(f).replace('_train_id.png', '.jpg'))
            for f in self.masks
        ]

        print(f"Found {len(self.images)} images and {len(self.masks)} masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.images[idx]).convert('RGB')
        orig_size = img.size  # (W, H)

        # Load pre-rasterized mask (already in Cityscapes train_id format)
        mask = Image.open(self.masks[idx])

        # Resize image & mask to target_size
        img_resized = img.resize(self.target_size, Image.BILINEAR)
        img_arr     = np.array(img_resized)

        # Random horizontal flip
        do_flip = self.flip_augment and random.random() > 0.5
        if do_flip:
            img_arr = img_arr[:, ::-1, :].copy()

        sam_img     = self.sam_transform.apply_image(img_arr)
        img_t       = torch.from_numpy(sam_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        mask = mask.resize(self.target_size, Image.NEAREST)
        mask_arr = np.array(mask).astype(np.int64)
        if do_flip:
            mask_arr = mask_arr[:, ::-1].copy()
        # Map 255 (BDD100K ignore) -> 19 (this project's ignore_index)
        mask_arr[mask_arr == 255] = self.ignore_index
        mask_t = torch.from_numpy(mask_arr).long()

        return {"image": img_t, "original_size": orig_size}, mask_t
