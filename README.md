# AD-SAM: Dual Encoder with Progressive Decoder for Semantic Segmentation

AD-SAM adapts the Segment Anything Model (SAM) for dense semantic segmentation on autonomous driving datasets. It pairs SAM's frozen ViT-H image encoder with a trainable ResNet-50 encoder, fuses their features through deformable convolutions with channel attention, and progressively decodes them back to full resolution using skip connections. The result is a model that leverages SAM's powerful visual representations while learning domain-specific features for pixel-level classification across 19 driving-scene classes.

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
              [B, 19, 1024, 1024]
```

### Key Components

**Dual Encoder** (`models/dual_encoder.py`): The core architecture. SAM's ViT-H encoder extracts high-level features at 64x64 resolution while ResNet-50 provides multi-scale features at 4 different resolutions. SAM's weights are frozen; ResNet is trainable with a 10x lower learning rate.

**Deformable Feature Fusion** (`models/deformable.py`): Fuses SAM and ResNet features using modulated deformable convolutions. Deformable convolutions learn spatial offsets to handle geometric variations in driving scenes (varying object scales, perspective distortion). Channel attention (`models/attention.py`) re-weights the fused channels.

**Progressive Decoder** (`models/dual_encoder.py`): Four upsample blocks progressively recover spatial resolution from 64x64 to 1024x1024. Each block performs 2x bilinear upsampling, concatenates a ResNet skip connection (reduced via 1x1 conv), and refines with two deformable convolution layers. GroupNorm and GELU activation are used throughout.

**SAM Adapter** (`models/sam_adapter.py`): Wraps SAM's image encoder and prompt encoder for compatibility with the dual encoder pipeline.

## Semantic Classes

The model segments 19 classes following the Cityscapes label convention:

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|----|-------|
| 0 | Road | 5 | Pole | 10 | Sky | 15 | Bus |
| 1 | Sidewalk | 6 | Traffic Light | 11 | Person | 16 | Train |
| 2 | Building | 7 | Traffic Sign | 12 | Rider | 17 | Motorcycle |
| 3 | Wall | 8 | Vegetation | 13 | Car | 18 | Bicycle |
| 4 | Fence | 9 | Terrain | 14 | Truck | | |

## Installation

```bash
# Create conda environment
conda create -n adsam python=3.10 -y
conda activate adsam

# Install PyTorch (adjust CUDA version to match your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install numpy pillow opencv-python matplotlib tqdm scipy
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional: CRF post-processing
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

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
│       ├── train/
│       │   ├── 0000f77c-6257be58.jpg
│       │   └── ...
│       └── val/
│           └── ...
└── labels/
    └── sem_seg/
        ├── train/
        │   ├── 0000f77c-6257be58_train_id.png
        │   └── ...
        └── val/
            └── ...
```

### SAM Checkpoint

Download the SAM ViT-H checkpoint (~2.4 GB):

```bash
mkdir -p models/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/checkpoints/
```

### Configure Paths

Edit `configs/paths.py` to point to your dataset locations and SAM checkpoint:

```python
CITYSCAPES_ROOT = "/path/to/Cityscapes"
BDD100K_ROOT = "/path/to/BDD100k"
SAM_CHECKPOINT_PATH = "/path/to/sam_vit_h_4b8939.pth"
```

Or set environment variables (these override the defaults in `configs/paths.py`):

```bash
export CITYSCAPES_ROOT=/path/to/Cityscapes
export BDD100K_ROOT=/path/to/BDD100k
export SAM_CHECKPOINT_PATH=/path/to/sam_vit_h_4b8939.pth
```

## Training

### Basic Usage

```bash
# Train on Cityscapes
python train.py --dataset_name cityscapes

# Train on BDD100K
python train.py --dataset_name bdd100k

# Quick test with limited samples
python train.py --dataset_name cityscapes --max_samples 100 --batch_size 4
```

### All Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | `cityscapes` | Dataset to use: `cityscapes` or `bdd100k` |
| `--max_samples` | int | `100` | Max number of training samples (use `None` for full dataset) |
| `--num_epochs` | int | `100` | Number of training epochs |
| `--batch_size` | int | `2` | Batch size for dataloaders |
| `--loss` | str | `hybrid` | Loss function: `hybrid` or `focaldice` |
| `--apply_crf` | flag | `False` | Apply DenseCRF post-processing during validation |
| `--gpu` | int | `0` | GPU device index (0, 1, or 2) |

### Training Configuration Details

- **Optimizer**: AdamW with weight decay 0.0005
- **Learning rates**: 2e-4 for decoder and fusion layers, 2e-5 for ResNet encoder
- **Scheduler**: Cosine annealing to 1e-6
- **Mixed precision**: Automatic via `torch.amp` (FP16 forward pass, FP32 gradients)
- **Input size**: All images resized to 1024x1024 with ImageNet normalization

### Multi-GPU

To select a specific GPU:

```bash
# Use GPU index directly
python train.py --gpu 2

# Or use CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2 python train.py --gpu 0
```

### Memory Tips

With SAM ViT-H at 1024x1024, expect ~8-10 GB VRAM per sample during training. On a 48 GB GPU (e.g., A6000):

- `--batch_size 2`: ~20 GB VRAM
- `--batch_size 4`: ~35 GB VRAM

If you encounter OOM errors, try:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --batch_size 2
```

## Loss Functions

### Hybrid Loss (`--loss hybrid`)

A weighted combination of four losses designed for semantic segmentation:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Focal Loss | 0.4 | Handles class imbalance by down-weighting easy examples |
| Dice Loss | 0.3 | Optimizes region overlap directly (IoU-like objective) |
| Lovasz-Softmax | 0.2 | Directly optimizes the Jaccard index (mIoU surrogate) |
| Surface/Boundary Loss | 0.1 | Penalizes errors proportional to distance from class boundaries |

### Focal-Dice Loss (`--loss focaldice`)

A simpler alternative combining Focal Loss (weight 0.5) and Dice Loss (weight 0.5).

## Outputs

Each training run creates timestamped outputs to avoid overwriting:

```
AD-SAM/
├── dual_encoder_ADSAM_cityscapes_100_hybrid_crf_False_20260312_113500.pth   # Best model checkpoint
├── training_log_ADSAM_cityscapes_100_hybrid_crf_False_20260312_113500.txt   # Full training log
└── run_lim_hybridADSAM_cityscapes_100_hybrid_crf_False_20260312_113500/     # Visualization directory
    ├── epoch_1_sample_0.png
    ├── epoch_1_sample_1.png
    └── ...
```

- **Checkpoints** (`.pth`): Saved whenever validation mIoU improves. Contains model weights, optimizer state, epoch, and best mIoU.
- **Training logs** (`.txt`): Per-epoch train/val loss, mIoU, and per-class IoU.
- **Visualizations**: Side-by-side comparisons of input image, ground truth, and prediction (with optional CRF refinement) for 4 validation samples each epoch.

## CRF Post-Processing

Optional DenseCRF refinement can be applied during validation to sharpen predictions along object boundaries:

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
python train.py --apply_crf
```

The CRF uses a Gaussian spatial kernel and a bilateral (spatial + color) kernel to encourage label consistency among nearby pixels with similar colors.

## Project Structure

```
AD-SAM/
├── configs/
│   └── paths.py              # Dataset and checkpoint path configuration
├── data/
│   ├── bdd100k_dataset.py    # BDD100K dataset class
│   ├── cityscapes_dataset.py # Cityscapes dataset class
│   ├── dataloaders.py        # Unified dataloader factory
│   └── preprocessing.py      # Batch preprocessing utilities
├── losses/
│   ├── focal_dice.py         # Combined Focal + Dice loss
│   └── hybrid_loss.py        # Hybrid loss (Focal + Dice + Lovasz + Surface)
├── metrics/
│   └── iou_metric.py         # Confusion matrix-based mIoU computation
├── models/
│   ├── attention.py          # Channel attention (squeeze-and-excitation)
│   ├── deformable.py         # Deformable convolution and feature fusion
│   ├── dual_encoder.py       # Main model: DualEncoderDeformableDecoder
│   └── sam_adapter.py        # SAM model wrapper
├── utils/
│   ├── crf_postprocess.py    # DenseCRF post-processing
│   ├── setup.py              # SAM model initialization
│   └── visualize.py          # Prediction visualization
├── train.py                  # Training entry point
└── requirements.txt          # Python dependencies
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
