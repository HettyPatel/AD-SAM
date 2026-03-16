import os
import random

import numpy as np
import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset


class CityscapesCachedDataset(Dataset):
    """
    Cityscapes dataset that loads pre-generated SAM embeddings from disk.
    Returns SAM embeddings + preprocessed images (for ResNet) + masks.
    Supports random horizontal flip augmentation via pre-generated flipped embeddings.
    """

    def __init__(self, image_dir, mask_dir, embedding_dir,
                 target_size=(1024, 1024), max_samples=None,
                 flip_augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.embedding_dir = embedding_dir
        self.target_size = target_size
        self.flip_augment = flip_augment
        self.ignore_index = 19

        self.sam_transform = ResizeLongestSide(target_size[0])

        self.images = sorted([
            os.path.join(root, name)
            for root, _, files in os.walk(image_dir)
            for name in files
            if name.endswith('leftImg8bit.png')
        ])

        self.masks = sorted([
            os.path.join(root, name)
            for root, _, files in os.walk(mask_dir)
            for name in files
            if name.endswith('gtFine_labelIds.png')
        ])

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]

        print(f'Found {len(self.images)} images and {len(self.masks)} masks')
        self._validate_pairs()

        self.label_mapping = {
            0: self.ignore_index, 1: self.ignore_index, 2: self.ignore_index,
            3: self.ignore_index, 4: self.ignore_index, 5: self.ignore_index,
            6: self.ignore_index, 7: 0, 8: 1, 9: self.ignore_index,
            10: self.ignore_index, 11: 2, 12: 3, 13: 4, 14: self.ignore_index,
            15: self.ignore_index, 16: self.ignore_index, 17: 5,
            18: self.ignore_index, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: self.ignore_index,
            30: self.ignore_index, 31: 16, 32: 17, 33: 18,
        }

    def _validate_pairs(self):
        for img_path, mask_path in zip(self.images, self.masks):
            img_name = os.path.basename(img_path).replace('_leftImg8bit.png', '')
            mask_name = os.path.basename(mask_path).replace('_gtFine_labelIds.png', '')
            assert img_name == mask_name, f"Mismatch: {img_path} vs {mask_path}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Decide whether to flip
        do_flip = self.flip_augment and random.random() > 0.5

        # Load pre-generated SAM embedding
        basename = os.path.splitext(os.path.basename(self.images[idx]))[0]
        if do_flip:
            emb_path = os.path.join(self.embedding_dir, f"{basename}_flip.pt")
        else:
            emb_path = os.path.join(self.embedding_dir, f"{basename}.pt")
        sam_embedding = torch.load(emb_path, weights_only=True)  # [256, 64, 64]

        # Load and preprocess image for ResNet
        image = Image.open(self.images[idx]).convert('RGB')
        original_size = image.size
        image = image.resize((1024, 1024), Image.BILINEAR)
        image_np = np.array(image)
        if do_flip:
            image_np = image_np[:, ::-1, :].copy()
        input_image = self.sam_transform.apply_image(image_np)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_image = (input_image - mean) / std

        # Process mask
        mask = Image.open(self.masks[idx])
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask = np.array(mask)
        if do_flip:
            mask = mask[:, ::-1].copy()
        mask = torch.from_numpy(mask).long()
        mask = torch.tensor([self.label_mapping.get(x.item(), self.ignore_index)
                             for x in mask.flatten()], dtype=torch.long)
        mask = mask.reshape(self.target_size)

        batched_input = {
            "image": input_image,
            "sam_embedding": sam_embedding,
            "original_size": original_size,
        }

        return batched_input, mask
