import os
import torch
from torch.utils.data import DataLoader
from .cityscapes_dataset import CityscapesDataset
from .bdd100k_dataset import BDD100kDataset
from .cityscapes_cached_dataset import CityscapesCachedDataset
from .bdd100k_cached_dataset import BDD100kCachedDataset

_DEFAULT_WORKERS = 0 if os.name == "nt" else 12


def get_dataloader(dataset_name, image_dir, mask_dir, batch_size=2, num_workers=_DEFAULT_WORKERS,
                   max_samples=None, target_size=(1024, 1024),
                   embedding_dir=None, flip_augment=False, is_train=True, native_mask=False):
    """
    is_train:    shuffle and drop the last incomplete batch (training only).
    native_mask: also return the label map at its original resolution under
                 batch_input["mask_native"] (used by --eval_native).
    """
    name = dataset_name.lower()
    common = dict(max_samples=max_samples, target_size=target_size,
                  flip_augment=flip_augment, native_mask=native_mask)
    if embedding_dir is not None:
        if name == 'cityscapes':
            dataset = CityscapesCachedDataset(image_dir=image_dir, mask_dir=mask_dir,
                                              embedding_dir=embedding_dir, **common)
        elif name == 'bdd100k':
            dataset = BDD100kCachedDataset(image_dir=image_dir, annot_dir=mask_dir,
                                           embedding_dir=embedding_dir, **common)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    else:
        if name == 'cityscapes':
            dataset = CityscapesDataset(image_dir=image_dir, mask_dir=mask_dir, **common)
        elif name == 'bdd100k':
            dataset = BDD100kDataset(image_dir=image_dir, annot_dir=mask_dir, **common)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
