"""Dataset and checkpoint locations. Override with environment variables:

    CITYSCAPES_ROOT      .../Cityscapes   (leftImg8bit/{train,val}, gtFine/{train,val})
    BDD100K_ROOT         .../BDD100K      (images/10k/{train,val}, labels/sem_seg/{train,val})
    SAM_CHECKPOINT_PATH  .../sam_vit_h_4b8939.pth
"""
import os

CITYSCAPES_ROOT = os.environ.get("CITYSCAPES_ROOT", "data/Cityscapes")
BDD100K_ROOT = os.environ.get("BDD100K_ROOT", "data/BDD100K")
SAM_CHECKPOINT_PATH = os.environ.get("SAM_CHECKPOINT_PATH", "checkpoints/sam_vit_h_4b8939.pth")

CITYSCAPES_TRAIN_IMAGES = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", "train")
CITYSCAPES_TRAIN_MASKS = os.path.join(CITYSCAPES_ROOT, "gtFine", "train")
CITYSCAPES_VAL_IMAGES = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", "val")
CITYSCAPES_VAL_MASKS = os.path.join(CITYSCAPES_ROOT, "gtFine", "val")

BDD100K_TRAIN_IMAGES = os.path.join(BDD100K_ROOT, "images", "10k", "train")
BDD100K_TRAIN_MASKS = os.path.join(BDD100K_ROOT, "labels", "sem_seg", "train")
BDD100K_VAL_IMAGES = os.path.join(BDD100K_ROOT, "images", "10k", "val")
BDD100K_VAL_MASKS = os.path.join(BDD100K_ROOT, "labels", "sem_seg", "val")


def dataset_paths(name):
    """(train_images, train_masks, val_images, val_masks) for 'cityscapes' or 'bdd100k'."""
    name = name.lower()
    if name == "cityscapes":
        return CITYSCAPES_TRAIN_IMAGES, CITYSCAPES_TRAIN_MASKS, CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_MASKS
    if name == "bdd100k":
        return BDD100K_TRAIN_IMAGES, BDD100K_TRAIN_MASKS, BDD100K_VAL_IMAGES, BDD100K_VAL_MASKS
    raise ValueError(f"Unsupported dataset: {name}")
