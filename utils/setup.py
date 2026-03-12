from segment_anything import sam_model_registry
from configs.paths import SAM_CHECKPOINT_PATH


def initialize_sam_model():
    model_type = 'vit_h'
    sam_model = sam_model_registry[model_type](checkpoint=None)
    
    return sam_model



