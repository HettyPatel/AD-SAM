import os
import random

import numpy as np
import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

from .cityscapes_dataset import (IGNORE_INDEX, CITYSCAPES_LABEL_MAPPING, build_label_lut,
                                 find_cityscapes_pairs, prepare_image)


class CityscapesCachedDataset(Dataset):
    """
    Cityscapes with pre-generated SAM embeddings loaded from disk.
    Returns ({"image", "sam_embedding", "original_size"[, "mask_native"]}, mask).
    Horizontal flip uses the pre-generated `<name>_flip.pt` embedding.
    """

    def __init__(self, image_dir, mask_dir, embedding_dir, target_size=(1024, 1024),
                 max_samples=None, flip_augment=True, native_mask=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.embedding_dir = embedding_dir
        self.target_size = target_size
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
        return torch.from_numpy(self._lut[arr])

    def __getitem__(self, idx):
        do_flip = self.flip_augment and random.random() > 0.5

        basename = os.path.splitext(os.path.basename(self.images[idx]))[0]
        emb_name = f"{basename}_flip.pt" if do_flip else f"{basename}.pt"
        sam_embedding = torch.load(os.path.join(self.embedding_dir, emb_name), weights_only=True)

        image = Image.open(self.images[idx]).convert('RGB')
        mask_pil = Image.open(self.masks[idx])
        original_size = image.size
        input_image = prepare_image(image, self.target_size, self.sam_transform, do_flip)
        mask = self._map_mask(mask_pil.resize(self.target_size, Image.NEAREST), do_flip)

        batched_input = {"image": input_image, "sam_embedding": sam_embedding,
                         "original_size": original_size}
        if self.native_mask:
            batched_input["mask_native"] = self._map_mask(mask_pil, do_flip)
        return batched_input, mask
