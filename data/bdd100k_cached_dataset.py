import os
import random

import numpy as np
import torch
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset


class BDD100kCachedDataset(Dataset):
    """
    BDD100K dataset that loads pre-generated SAM embeddings from disk.
    Returns SAM embeddings + preprocessed images (for ResNet) + masks.
    Supports random horizontal flip augmentation via pre-generated flipped embeddings.
    """

    def __init__(self, image_dir, annot_dir, embedding_dir,
                 target_size=(1024, 1024), max_samples=None,
                 flip_augment=True):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.embedding_dir = embedding_dir
        self.target_size = target_size
        self.flip_augment = flip_augment
        self.ignore_index = 19

        self.sam_transform = ResizeLongestSide(target_size[0])

        self.masks = sorted([
            os.path.join(annot_dir, f)
            for f in os.listdir(annot_dir)
            if f.endswith('_train_id.png')
        ])
        if max_samples is not None:
            self.masks = self.masks[:max_samples]

        self.images = [
            os.path.join(image_dir, os.path.basename(f).replace('_train_id.png', '.jpg'))
            for f in self.masks
        ]

        print(f"Found {len(self.images)} images and {len(self.masks)} masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        do_flip = self.flip_augment and random.random() > 0.5

        # Load pre-generated SAM embedding
        # BDD100K image basenames end in .jpg, embedding basenames match without extension
        basename = os.path.splitext(os.path.basename(self.images[idx]))[0]
        if do_flip:
            emb_path = os.path.join(self.embedding_dir, f"{basename}_flip.pt")
        else:
            emb_path = os.path.join(self.embedding_dir, f"{basename}.pt")
        sam_embedding = torch.load(emb_path, weights_only=True)  # [256, 64, 64]

        # Load and preprocess image for ResNet
        img = Image.open(self.images[idx]).convert('RGB')
        orig_size = img.size
        img_resized = img.resize(self.target_size, Image.BILINEAR)
        img_arr = np.array(img_resized)
        if do_flip:
            img_arr = img_arr[:, ::-1, :].copy()
        sam_img = self.sam_transform.apply_image(img_arr)
        img_t = torch.from_numpy(sam_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        # Process mask
        mask = Image.open(self.masks[idx])
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask_arr = np.array(mask).astype(np.int64)
        if do_flip:
            mask_arr = mask_arr[:, ::-1].copy()
        mask_arr[mask_arr == 255] = self.ignore_index
        mask_t = torch.from_numpy(mask_arr).long()

        return {"image": img_t, "sam_embedding": sam_embedding, "original_size": orig_size}, mask_t
