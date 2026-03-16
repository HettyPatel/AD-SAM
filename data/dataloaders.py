import os
import torch
from torch.utils.data import DataLoader
from .cityscapes_dataset import CityscapesDataset
from .bdd100k_dataset import BDD100kDataset
from .cityscapes_cached_dataset import CityscapesCachedDataset
from .bdd100k_cached_dataset import BDD100kCachedDataset

_DEFAULT_WORKERS = 0 if os.name == "nt" else 4

def get_dataloader(dataset_name, image_dir, mask_dir, batch_size=2, num_workers=_DEFAULT_WORKERS, max_samples=None,target_size=(1024, 1024),
                   embedding_dir=None, flip_augment=False):
    if embedding_dir is not None:
        # Use cached SAM embeddings
        if dataset_name.lower() == 'cityscapes':
            dataset = CityscapesCachedDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                embedding_dir=embedding_dir,
                max_samples=max_samples,
                target_size=target_size,
                flip_augment=flip_augment,
            )
        elif dataset_name.lower() == 'bdd100k':
            dataset = BDD100kCachedDataset(
                image_dir=image_dir,
                annot_dir=mask_dir,
                embedding_dir=embedding_dir,
                max_samples=max_samples,
                target_size=target_size,
                flip_augment=flip_augment,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    else:
        # Original: load raw images (SAM encoder runs during training)
        if dataset_name.lower() == 'cityscapes':
            dataset = CityscapesDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                max_samples=max_samples,
                target_size=target_size,
                flip_augment=flip_augment,
            )
        elif dataset_name.lower() == 'bdd100k':
            dataset = BDD100kDataset(
                image_dir=image_dir,
                annot_dir=mask_dir,
                max_samples=max_samples,
                target_size=target_size,
                flip_augment=flip_augment,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader
