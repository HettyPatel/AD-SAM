from segment_anything.modeling import Sam
import torch.nn as nn
import torch.nn.functional as F

class MulticlassSam(nn.Module):
    def __init__(self, sam_model: Sam, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        
        # Keep original SAM components
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        #original_decoder = sam_model.mask_decoder
        
    def forward(self, batched_input, multimask_output=False):
        # Process input image
        image_embeddings = self.image_encoder(batched_input["image"])
        
        # Process prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        # Generate masks
        masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        return masks

class SemanticDecoder(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        
        # SAM's image encoder output dimensions
        self.encoder_dim = 256
        self.encoder_h = 64  #
        self.encoder_w = 64
        
        # Decoder architecture
        self.conv1 = nn.Conv2d(self.encoder_dim, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, image_embeddings):
        # Apply convolution blocks with skip connections
        x = F.relu(self.bn1(self.conv1(image_embeddings)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        return x