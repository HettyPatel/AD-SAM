# AD-SAM: Dual Encoder with Progressive Decoder for Semantic Segmentation

AD-SAM adapts the Segment Anything Model (SAM) for dense semantic segmentation. It pairs SAM's frozen ViT-H image encoder with a trainable ResNet-50 encoder, fuses their features through deformable convolutions with channel attention, and progressively decodes them back to full resolution using skip connections. The result is a model that leverages SAM's powerful visual representations while learning domain-specific features for pixel-level classification.

## Architecture

```
Input Image (1024x1024)
        |
        +---------------------------+
        |                           |
  SAM ViT-H Encoder          ResNet-50 Encoder
    (frozen)                   (trainable)
  [B, 256, 64, 64]           layer1: [B,  256, 256, 256]
                              layer2: [B,  512, 128, 128]
                              layer3: [B, 1024,  64,  64]
                              layer4: [B, 2048,  32,  32]
        |                           |
        +---------- Fusion ---------+
                      |
          DeformableFeatureFusion
           (concat + deformable conv
            + channel attention)
              [B, 256, 64, 64]
                      |
              Progressive Decoder
                      |
          UpsampleBlock 1 (+ ResNet layer3 skip) -> [B, 128, 128, 128]
          UpsampleBlock 2 (+ ResNet layer2 skip) -> [B,  64, 256, 256]
          UpsampleBlock 3 (+ ResNet layer1 skip) -> [B,  32, 512, 512]
          UpsampleBlock 4 (no skip)              -> [B,  32, 1024, 1024]
                      |
              Classification Head
              [B, num_classes, 1024, 1024]
```

### Key Components

**Dual Encoder** (`models/dual_encoder.py`): The core architecture. SAM's ViT-H encoder extracts high-level features at 64x64 resolution while ResNet-50 provides multi-scale features at 4 different resolutions. SAM's weights are frozen; ResNet is trainable with a 10x lower learning rate.

**Deformable Feature Fusion** (`models/deformable.py`): Fuses SAM and ResNet features using modulated deformable convolutions. Deformable convolutions learn spatial offsets to handle geometric variations in driving scenes (varying object scales, perspective distortion). Channel attention (`models/attention.py`) re-weights the fused channels.

**Progressive Decoder** (`models/dual_encoder.py`): Four upsample blocks progressively recover spatial resolution from 64x64 to 1024x1024. Each block performs 2x bilinear upsampling, concatenates a ResNet skip connection (reduced via 1x1 conv), and refines with two deformable convolution layers. GroupNorm and GELU activation are used throughout.

**SAM Adapter** (`models/sam_adapter.py`): Wraps SAM's image encoder and prompt encoder for compatibility with the dual encoder pipeline.

## Installation

```bash
# Create conda environment
conda create -n adsam python=3.10 -y
conda activate adsam

# Install PyTorch (adjust CUDA version to match your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install numpy pillow opencv-python matplotlib tqdm scipy

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional: CRF post-processing
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Quick Start

### 1. Download the SAM checkpoint

```bash
mkdir -p models/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/checkpoints/
```

### 2. Configure paths

Edit `configs/paths.py` or set environment variables:

```bash
export CITYSCAPES_ROOT=/path/to/Cityscapes
export BDD100K_ROOT=/path/to/BDD100k
export SAM_CHECKPOINT_PATH=/path/to/sam_vit_h_4b8939.pth
```

### 3. Pre-generate SAM embeddings (recommended)

Since the SAM encoder is frozen, you can pre-compute its embeddings once to significantly speed up training. This generates both original and horizontally flipped variants for data augmentation:

```bash
python pregenerate_embeddings.py --dataset_name cityscapes --gpu 0
python pregenerate_embeddings.py --dataset_name bdd100k --gpu 0
```

Embeddings are saved to `embeddings/` by default (~4 MB per image, ~24 GB for Cityscapes, ~57 GB for BDD100K, doubled for flipped variants). You can change the output location with `--output_root /path/to/embeddings`.

### 4. Train

```bash
# Train with pre-generated embeddings + flip augmentation (fast)
python train.py --dataset_name cityscapes --embedding_dir embeddings --flip_augment

# Train without embeddings (slower, encoder runs each epoch)
python train.py --dataset_name cityscapes

# Quick test with limited samples
python train.py --dataset_name cityscapes --max_samples 100 --embedding_dir embeddings --flip_augment
```

## Using a Custom Dataset

AD-SAM can be adapted to any semantic segmentation dataset. Here's what you need to do:

### 1. Create a dataset class

Create a new file in `data/` (e.g., `data/my_dataset.py`). Your dataset must return a dictionary and a mask tensor:

```python
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(1024, 1024),
                 max_samples=None, flip_augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.flip_augment = flip_augment
        self.num_classes = 10  # <-- your number of classes
        self.ignore_index = 10  # <-- index for unlabeled/ignore pixels

        self.sam_transform = ResizeLongestSide(target_size[0])

        # Collect your image and mask file paths (sorted, matched)
        self.images = sorted([...])
        self.masks = sorted([...])

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        original_size = image.size

        # Resize to 1024x1024
        image = image.resize((1024, 1024), Image.BILINEAR)
        image_np = np.array(image)

        # Optional flip
        do_flip = self.flip_augment and random.random() > 0.5
        if do_flip:
            image_np = image_np[:, ::-1, :].copy()

        # ImageNet normalization (used by ResNet branch)
        input_image = self.sam_transform.apply_image(image_np)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_image = (input_image - mean) / std

        # Process mask - must be a LongTensor with values in [0, num_classes-1]
        # and ignore_index for unlabeled pixels
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask_np = np.array(mask).astype(np.int64)
        if do_flip:
            mask_np = mask_np[:, ::-1].copy()
        # Map your raw label IDs to contiguous class indices here if needed
        mask_t = torch.from_numpy(mask_np).long()

        return {"image": input_image, "original_size": original_size}, mask_t
```

For cached embeddings, create a corresponding cached dataset class (see `data/cityscapes_cached_dataset.py` as a template) that also loads `sam_embedding` from a `.pt` file and includes it in the returned dictionary.

### 2. Register your dataset in `data/dataloaders.py`

Add your dataset to the `get_dataloader()` function:

```python
from .my_dataset import MyDataset

# Inside get_dataloader(), add an elif branch:
elif dataset_name.lower() == 'mydataset':
    dataset = MyDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        max_samples=max_samples,
        target_size=target_size,
        flip_augment=flip_augment,
    )
```

### 3. Update `configs/paths.py`

Add paths for your dataset:

```python
MYDATASET_ROOT = os.environ.get("MYDATASET_ROOT", "/path/to/mydataset")
MYDATASET_TRAIN_IMAGES = os.path.join(MYDATASET_ROOT, "train", "images")
MYDATASET_TRAIN_MASKS  = os.path.join(MYDATASET_ROOT, "train", "masks")
MYDATASET_VAL_IMAGES   = os.path.join(MYDATASET_ROOT, "val", "images")
MYDATASET_VAL_MASKS    = os.path.join(MYDATASET_ROOT, "val", "masks")
```

### 4. Update `train.py`

Add your dataset paths and change `num_classes`:

- Add an `elif` in the `__main__` block for your dataset name
- Change `num_classes=19` to your number of classes throughout (in `build_ablation_model`, `IoUMetric`, loss functions, and `CLASS_NAMES`)
- Update `ignore_index` if yours differs from 19

### 5. Update `pregenerate_embeddings.py`

Add your dataset to `get_image_paths()` so you can pre-generate embeddings:

```python
elif dataset_name == 'mydataset':
    if split == 'train':
        img_dir = MYDATASET_TRAIN_IMAGES
    else:
        img_dir = MYDATASET_VAL_IMAGES
    return sorted([
        os.path.join(root, name)
        for root, _, files in os.walk(img_dir)
        for name in files
        if name.endswith('.png')  # adjust to your file extension
    ])
```

### Key things to get right

- **Mask values**: Must be contiguous integers `[0, 1, ..., num_classes-1]` with `ignore_index` for unlabeled pixels. If your masks use different IDs, add a label mapping (see `cityscapes_dataset.py` for an example).
- **`num_classes`**: Must match everywhere — model, loss, metrics, dataset.
- **`ignore_index`**: Must match between dataset, loss function (`ignore_index=N`), and metrics.
- **Image size**: Always 1024x1024 (SAM's expected input).

## Dataset Setup

### Cityscapes

1. Register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
2. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`
3. Extract to your dataset directory

Expected structure:
```
Cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
└── gtFine/
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019_gtFine_labelIds.png
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...
```

### BDD100K

1. Register at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/)
2. Download the **10K images** and **semantic segmentation labels**
3. Extract and organize as shown below

Expected structure:
```
BDD100k/
├── images/
│   └── 10k/
│       ├── train/   (7000 images)
│       │   ├── 0000f77c-6257be58.jpg
│       │   └── ...
│       └── val/     (1000 images)
│           └── ...
└── labels/
    └── sem_seg/
        ├── train/
        │   ├── 0000f77c-6257be58_train_id.png
        │   └── ...
        └── val/
            └── ...
```

## Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | `cityscapes` | Dataset: `cityscapes` or `bdd100k` |
| `--max_samples` | int/None | `100` | Max training samples (`None` for full dataset) |
| `--num_epochs` | int | `100` | Number of training epochs |
| `--batch_size` | int | `2` | Batch size |
| `--loss` | str | `hybrid` | Loss function: `hybrid` or `focaldice` |
| `--ablation` | str | `full` | Ablation variant: `full`, `no_deform`, `no_attention`, `sam_encoder_only`, `ce_loss` |
| `--apply_crf` | flag | `False` | Apply DenseCRF post-processing during validation |
| `--embedding_dir` | str | None | Path to pre-generated SAM embeddings |
| `--flip_augment` | flag | `False` | Enable 50% random horizontal flip augmentation |
| `--gpu` | int | `0` | GPU device index |
| `--vis_every` | int | `1` | Visualize every N epochs (0 to disable) |

## Training Details

- **Optimizer**: AdamW with weight decay 0.0005
- **Learning rates**: 2e-4 for decoder and fusion layers, 2e-5 for ResNet encoder
- **Scheduler**: Cosine annealing to 1e-6
- **Mixed precision**: Automatic via `torch.amp` (FP16 forward pass, FP32 gradients)
- **Input size**: All images resized to 1024x1024
- **Normalization**: ImageNet normalization for ResNet branch; SAM-specific normalization for SAM encoder (handled automatically)
- **Augmentation**: Random horizontal flip (50%) when `--flip_augment` is enabled

## Loss Functions

### Hybrid Loss (`--loss hybrid`)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Focal Loss | 0.4 | Handles class imbalance by down-weighting easy examples |
| Dice Loss | 0.3 | Optimizes region overlap directly (IoU-like objective) |
| Lovasz-Softmax | 0.2 | Directly optimizes the Jaccard index (mIoU surrogate) |
| Surface/Boundary Loss | 0.1 | Penalizes errors proportional to distance from class boundaries |

### Focal-Dice Loss (`--loss focaldice`)

Simpler alternative: Focal Loss (0.5) + Dice Loss (0.5).

## Ablation Study

Run ablation variants to measure the contribution of each component:

```bash
# Full model (default)
python train.py --ablation full --embedding_dir embeddings --flip_augment

# No deformable convolutions (standard conv, keeps attention)
python train.py --ablation no_deform --embedding_dir embeddings --flip_augment

# No channel attention (keeps deformable conv)
python train.py --ablation no_attention --embedding_dir embeddings --flip_augment

# SAM encoder only (no ResNet branch)
python train.py --ablation sam_encoder_only --embedding_dir embeddings --flip_augment

# Cross-entropy loss instead of hybrid
python train.py --ablation ce_loss --embedding_dir embeddings --flip_augment
```

## Running All Experiments

The `run_overnight.sh` script runs the full experiment suite:

```bash
nohup bash run_overnight.sh > overnight_output.log 2>&1 &
tail -f overnight_output.log
```

This runs:
1. **Group 0**: Pre-generate SAM embeddings for both datasets
2. **Group 1**: AD-SAM data efficiency (100, 500, 1000, full samples on Cityscapes and BDD100K)
3. **Group 2**: DeepLabV3 baseline at the same data sizes
4. **Group 3**: Ablation study on Cityscapes full
5. **Group 4**: CRF post-processing evaluation
6. **Group 5**: Efficiency metrics (params/FLOPs/latency)

## Outputs

Each training run creates timestamped outputs:

```
AD-SAM/
├── dual_encoder_ADSAM_cityscapes_100_hybrid_crf_False_full_<timestamp>.pth  # Best checkpoint
├── training_log_ADSAM_cityscapes_100_hybrid_crf_False_full_<timestamp>.txt  # Training log
├── results/
│   ├── curves_ADSAM_cityscapes_100_hybrid_crf_False_full_<timestamp>.csv    # Per-epoch metrics
│   └── results_summary.csv                                                  # Aggregated results
└── run_lim_hybrid.../                                                       # Visualizations
```

## Memory Tips

With SAM ViT-H at 1024x1024, expect ~8-10 GB VRAM per sample. On a 48 GB GPU (e.g., A6000):

- `--batch_size 2`: ~20 GB VRAM
- `--batch_size 4`: ~35 GB VRAM

Using pre-generated embeddings (`--embedding_dir`) reduces VRAM since the SAM encoder doesn't run during training.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --batch_size 2
```

## Project Structure

```
AD-SAM/
├── configs/
│   └── paths.py                    # Dataset and checkpoint path configuration
├── data/
│   ├── bdd100k_dataset.py          # BDD100K dataset class
│   ├── bdd100k_cached_dataset.py   # BDD100K with pre-generated SAM embeddings
│   ├── cityscapes_dataset.py       # Cityscapes dataset class
│   ├── cityscapes_cached_dataset.py# Cityscapes with pre-generated SAM embeddings
│   ├── dataloaders.py              # Unified dataloader factory
│   └── preprocessing.py            # Batch preprocessing utilities
├── losses/
│   ├── focal_dice.py               # Combined Focal + Dice loss
│   └── hybrid_loss.py              # Hybrid loss (Focal + Dice + Lovasz + Surface)
├── metrics/
│   └── iou_metric.py               # Confusion matrix-based mIoU computation
├── models/
│   ├── attention.py                # Channel attention (squeeze-and-excitation)
│   ├── deformable.py               # Deformable convolution and feature fusion
│   ├── dual_encoder.py             # Main model: DualEncoderDeformableDecoder
│   └── sam_adapter.py              # SAM model wrapper
├── utils/
│   ├── crf_postprocess.py          # DenseCRF post-processing
│   ├── setup.py                    # SAM model initialization
│   └── visualize.py                # Prediction visualization
├── pregenerate_embeddings.py       # Pre-generate SAM encoder embeddings
├── train.py                        # AD-SAM training entry point
├── train_deeplabv3.py              # DeepLabV3 baseline training
├── run_ablations.py                # Ablation model variants
├── run_overnight.sh                # Full experiment suite
├── collect_results.py              # Aggregate results from logs
└── compute_efficiency.py           # Parameter/FLOP/latency analysis
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{adsam2025,
  title={AD-SAM: Dual Encoder with Progressive Decoder for Semantic Segmentation},
  author={},
  year={2025}
}
```
