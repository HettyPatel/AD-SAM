import os

# ── Set these to YOUR local dataset locations ──
# Override via environment variables: CITYSCAPES_ROOT, BDD100K_ROOT, SAM_CHECKPOINT_PATH

CITYSCAPES_ROOT = os.environ.get(
    "CITYSCAPES_ROOT",
    "/data/hpate061/Data/Cityscapes"
)

BDD100K_ROOT = os.environ.get(
    "BDD100K_ROOT",
    "/data/home/hpate061/datasets/BDD100k_poly"
)

CITYSCAPES_TRAIN_IMAGES = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", "train")
CITYSCAPES_TRAIN_MASKS  = os.path.join(CITYSCAPES_ROOT, "gtFine", "train")
CITYSCAPES_VAL_IMAGES   = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", "val")
CITYSCAPES_VAL_MASKS    = os.path.join(CITYSCAPES_ROOT, "gtFine", "val")

BDD100K_TRAIN_IMAGES = os.path.join(BDD100K_ROOT, "images", "10k", "train")
BDD100K_TRAIN_MASKS  = os.path.join(BDD100K_ROOT, "labels", "sem_seg", "train")
BDD100K_VAL_IMAGES   = os.path.join(BDD100K_ROOT, "images", "10k", "val")
BDD100K_VAL_MASKS    = os.path.join(BDD100K_ROOT, "labels", "sem_seg", "val")

SAM_CHECKPOINT_PATH = os.environ.get(
    "SAM_CHECKPOINT_PATH",
    "/data/home/hpate061/models/checkpoints/sam_vit_h_4b8939.pth"
)