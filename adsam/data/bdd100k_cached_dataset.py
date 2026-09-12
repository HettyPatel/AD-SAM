import os
import random

import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

from .cityscapes_dataset import IGNORE_INDEX, prepare_image
from .bdd100k_dataset import find_bdd100k_pairs, map_bdd_mask


class BDD100kCachedDataset(Dataset):
    """
    BDD100K with pre-generated SAM embeddings loaded from disk.
    Returns ({"image", "sam_embedding", "original_size"[, "mask_native"]}, mask).
    """

    def __init__(self, image_dir, annot_dir, embedding_dir, target_size=(1024, 1024),
                 max_samples=None, flip_augment=True, native_mask=False):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.embedding_dir = embedding_dir
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
        do_flip = self.flip_augment and random.random() > 0.5

        basename = os.path.splitext(os.path.basename(self.images[idx]))[0]
        emb_name = f"{basename}_flip.pt" if do_flip else f"{basename}.pt"
        sam_embedding = torch.load(os.path.join(self.embedding_dir, emb_name), weights_only=True)

        img = Image.open(self.images[idx]).convert('RGB')
        mask_pil = Image.open(self.masks[idx])
        orig_size = img.size
        img_t = prepare_image(img, self.target_size, self.sam_transform, do_flip)
        mask_t = map_bdd_mask(mask_pil.resize(self.target_size, Image.NEAREST), do_flip)

        batched_input = {"image": img_t, "sam_embedding": sam_embedding, "original_size": orig_size}
        if self.native_mask:
            batched_input["mask_native"] = map_bdd_mask(mask_pil, do_flip)
        return batched_input, mask_t
