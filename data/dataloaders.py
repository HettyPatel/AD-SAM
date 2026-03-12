import os
import torch
from torch.utils.data import DataLoader
from .cityscapes_dataset import CityscapesDataset
from .bdd100k_dataset import BDD100kDataset

_DEFAULT_WORKERS = 0 if os.name == "nt" else 4

def get_dataloader(dataset_name, image_dir, mask_dir, batch_size=2, num_workers=_DEFAULT_WORKERS, max_samples=None,target_size=(1024, 1024)):
    if dataset_name.lower() == 'cityscapes':
        dataset = CityscapesDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            max_samples=max_samples,
            target_size=target_size
        )
    elif dataset_name.lower() == 'bdd100k':
        dataset = BDD100kDataset(
            image_dir=image_dir,
            annot_dir=mask_dir,  
            max_samples=max_samples,
            target_size=target_size
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