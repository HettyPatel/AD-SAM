import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from adsam.config import SAM_CHECKPOINT_PATH


def initialize_sam_model():
    model_type = 'vit_h'
    sam_model = sam_model_registry[model_type](checkpoint=None)
    return sam_model


def load_sam_weights(sam_model, checkpoint_path=SAM_CHECKPOINT_PATH):
    """Load the SAM checkpoint into sam_model (shape-compatible tensors only)."""
    state = torch.load(checkpoint_path, map_location='cpu')
    own = sam_model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    sam_model.load_state_dict(compatible, strict=False)
    print(f"SAM checkpoint loaded: {len(compatible)}/{len(own)} tensors from {checkpoint_path}")
    return sam_model


class CachedSAMStub:
    """Stand-in for the SAM model when embeddings are pre-computed.

    The image encoder is never called in cached mode, so an Identity keeps the
    637M-parameter ViT-H (and its 2.5 GB checkpoint) off the GPU entirely.
    """
    def __init__(self):
        self.image_encoder = nn.Identity()


def build_sam_for_training(cached: bool):
    if cached:
        print("Cached embeddings enabled: SAM encoder not built")
        return CachedSAMStub()
    return load_sam_weights(initialize_sam_model())
