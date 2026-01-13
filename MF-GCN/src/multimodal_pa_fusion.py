import math
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tvm
except Exception:
    tvm = None

try:
    import timm
except Exception:
    timm = None

try:
    from transformers import AutoModel, AutoImageProcessor
except Exception:
    AutoModel = None
    AutoImageProcessor = None

from .cnn_backbones import get_backbone


class ResNetMultiScale(nn.Module):
    """
    Thin wrapper around ResNet18 to expose multi-scale feature maps.
    Returns features after each residual stage: c1..c4.
    """

    def __init__(self, weights: Optional[str] = "imagenet", shared_backbone: Optional[nn.Module] = None):
        super().__init__()
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            if tvm is None:
                raise ImportError("torchvision is required for ResNet backbones")
            weights_enum = tvm.ResNet18_Weights.DEFAULT if weights == "imagenet" else None
            self.backbone = tvm.resnet18(weights=weights_enum)
            # Remove classifier head; we only need features
            self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        c1 = b.maxpool(x)

        c1 = b.layer1(c1)  # 1/4 spatial
        c2 = b.layer2(c1)  # 1/8
        c3 = b.layer3(c2)  # 1/16
        c4 = b.layer4(c3)  # 1/32
        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


class ViTMultiScale(nn.Module):
    """
    Multi-scale feature extractor for Vision Transformer models (MAE, DINO, i-JEPA).
    Extracts features from different transformer layers and reshapes to spatial format.
    """
    
    def __init__(self, backbone_name: str, shared_backbone: Optional[nn.Module] = None, in_channels: int = 3):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            # Get the base backbone model
            if self.backbone_name == "mae_vit":
                if timm is None:
                    raise ImportError("timm not available for MAE ViT backbone")
                base = timm.create_model('vit_base_patch16_224', pretrained=False)
                # ckpt_path = os.environ.get("MAE_CKPT", "").strip()
                ckpt_path = "/home/ec2-user/SageMaker/gcn-deployment/gcn/checkpoint-29.pth"
                if ckpt_path:
                    try:
                        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                        if isinstance(state, dict) and any(k in state for k in ["model", "state_dict", "teacher", "student"]):
                            for key in ["model", "state_dict", "student"]:
                                if key in state and isinstance(state[key], dict):
                                    state = state[key]
                                    break
                        if isinstance(state, dict):
                            base_state = base.state_dict()
                            filtered = {k: v for k, v in state.items() if k in base_state and base_state[k].shape == v.shape}
                            base.load_state_dict(filtered, strict=False)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load MAE checkpoint '{ckpt_path}': {e}")
                self.backbone = base
                
            elif self.backbone_name == "dino_vits8":
                if timm is None:
                    raise ImportError("timm not available for DINO backbone")
                self.backbone = timm.create_model('vit_small_patch8_224.dino', pretrained=True)
                
            elif self.backbone_name == "ijepa_vit":
                if AutoModel is None:
                    raise ImportError("transformers not available for i-JEPA backbone")
                model_name = os.environ.get("IJEPA_MODEL", "facebook/ijepa_vith14_1k")
                try:
                    self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to load i-JEPA model '{model_name}': {e}")
            else:
                raise ValueError(f"Unsupported ViT backbone: {backbone_name}")
        
        # Determine feature dimensions and patch size
        if self.backbone_name == "ijepa_vit":
            # For HuggingFace i-JEPA models
            embed_dim = getattr(self.backbone.config, 'embed_dim', None) or getattr(self.backbone.config, 'hidden_size', None)
            if embed_dim is None:
                raise RuntimeError("Cannot determine embed_dim for i-JEPA model")
            self.embed_dim = embed_dim
            # Assume patch size 16 for standard ViT models (adjust if needed)
            self.patch_size = 16
            self.num_layers = getattr(self.backbone.config, 'num_hidden_layers', 12)
        else:
            # For timm models
            self.embed_dim = getattr(self.backbone, 'embed_dim', 768)
            self.patch_size = getattr(self.backbone, 'patch_embed', None)
            if self.patch_size is not None:
                self.patch_size = getattr(self.patch_size, 'patch_size', [16, 16])[0]
            else:
                self.patch_size = 16
            if hasattr(self.backbone, 'blocks'):
                self.num_layers = len(self.backbone.blocks)
            else:
                self.num_layers = 12
        
        # Select layers for multi-scale features (early, mid, late, final)
        self.layer_indices = [
            max(0, self.num_layers // 4 - 1),
            max(0, self.num_layers // 2 - 1),
            max(0, 3 * self.num_layers // 4 - 1),
            self.num_layers - 1
        ]
        
    def _extract_vit_features(self, x: torch.Tensor) -> dict:
        """Extract multi-scale features from ViT model"""
        B = x.shape[0]
        
        if self.backbone_name == "ijepa_vit":
            # For HuggingFace models, use forward with output_hidden_states
            outputs = self.backbone(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of (B, N, D)
            
            # Get features at selected layers
            features = {}
            img_size = int(x.shape[-1] / self.patch_size)  # Assuming square images
            
            for idx, layer_idx in enumerate(self.layer_indices):
                if layer_idx < len(hidden_states):
                    feat = hidden_states[layer_idx]  # (B, N, D)
                    # Remove CLS token if present (first token)
                    if feat.shape[1] > img_size * img_size:
                        feat = feat[:, 1:, :]  # Remove CLS token
                    # Reshape to spatial: (B, D, H, W)
                    feat = feat.reshape(B, img_size, img_size, -1).permute(0, 3, 1, 2)
                    features[f"c{idx+1}"] = feat
                else:
                    # Use last available layer
                    feat = hidden_states[-1]
                    if feat.shape[1] > img_size * img_size:
                        feat = feat[:, 1:, :]
                    feat = feat.reshape(B, img_size, img_size, -1).permute(0, 3, 1, 2)
                    features[f"c{idx+1}"] = feat
        else:
            # For timm models (MAE, DINO)
            # Manual forward through blocks to capture intermediate features
            x_emb = self.backbone.patch_embed(x)
            if hasattr(self.backbone, 'cls_token'):
                cls_token = self.backbone.cls_token.expand(B, -1, -1)
                x_emb = torch.cat([cls_token, x_emb], dim=1)
            if hasattr(self.backbone, 'pos_embed'):
                x_emb = x_emb + self.backbone.pos_embed
            if hasattr(self.backbone, 'pos_drop'):
                x_emb = self.backbone.pos_drop(x_emb)
            
            # Store intermediate features at selected layers
            hidden_states = {}
            if hasattr(self.backbone, 'blocks'):
                for i, blk in enumerate(self.backbone.blocks):
                    x_emb = blk(x_emb)
                    if i in self.layer_indices:
                        hidden_states[i] = x_emb
            else:
                # Fallback: use forward_features
                feat = self.backbone.forward_features(x)
                if isinstance(feat, dict):
                    feat = feat.get('x', None) or feat.get('features', None)
                if feat is not None:
                    # Use same feature for all scales
                    for layer_idx in self.layer_indices:
                        hidden_states[layer_idx] = feat
            
            # Extract features at selected layers
            features = {}
            num_patches = int((x.shape[-1] / self.patch_size) ** 2)  # Use actual input size
            img_size = int(x.shape[-1] / self.patch_size)
            
            for idx, layer_idx in enumerate(self.layer_indices):
                if layer_idx in hidden_states:
                    feat = hidden_states[layer_idx]  # (B, N, D)
                elif len(hidden_states) > 0:
                    # Use last available layer
                    feat = list(hidden_states.values())[-1]
                else:
                    # Final fallback: use forward_features
                    feat = self.backbone.forward_features(x)
                    if isinstance(feat, dict):
                        feat = feat.get('x', None) or feat.get('features', None)
                    if feat is None:
                        raise RuntimeError(f"Could not extract features from {self.backbone_name}")
                
                # Remove CLS token if present
                if feat.shape[1] > num_patches:
                    feat = feat[:, 1:, :]
                
                # Reshape to spatial: (B, D, H, W)
                if feat.shape[1] == num_patches:
                    feat = feat.reshape(B, img_size, img_size, -1).permute(0, 3, 1, 2)
                else:
                    # If shape doesn't match, try to infer spatial dimensions
                    sqrt_patches = int(feat.shape[1] ** 0.5)
                    if sqrt_patches * sqrt_patches == feat.shape[1]:
                        feat = feat.reshape(B, sqrt_patches, sqrt_patches, -1).permute(0, 3, 1, 2)
                        # Interpolate to target size if needed
                        if sqrt_patches != img_size:
                            feat = F.interpolate(feat, size=(img_size, img_size), mode='bilinear', align_corners=False)
                    else:
                        # Fallback: use adaptive pooling
                        feat = feat.mean(dim=1, keepdim=True)  # (B, 1, D)
                        feat = feat.unsqueeze(-1).expand(-1, -1, -1, img_size, img_size)  # (B, 1, D, H, W)
                        feat = feat.squeeze(1)  # (B, D, H, W)
                
                features[f"c{idx+1}"] = feat
        
        return features
    
    def forward(self, x: torch.Tensor) -> dict:
        return self._extract_vit_features(x)


def create_multiscale_backbone(backbone_name: str, shared_backbone: Optional[nn.Module] = None, in_channels: int = 3):
    """
    Factory function to create a multi-scale backbone extractor.
    
    Args:
        backbone_name: Name of the backbone ("resnet18", "mae_vit", "dino_vits8", "ijepa_vit")
        shared_backbone: Optional shared backbone module for weight sharing
        in_channels: Number of input channels
    
    Returns:
        Multi-scale backbone extractor
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name == "resnet18":
        return ResNetMultiScale(weights="imagenet", shared_backbone=shared_backbone)
    elif backbone_name in ["mae_vit", "dino_vits8", "ijepa_vit"]:
        return ViTMultiScale(backbone_name, shared_backbone=shared_backbone, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported backbone for multi-scale extraction: {backbone_name}")


class PA3WayBlock(nn.Module):
    """
    Parallel attention fusion for three modalities at a single scale.
    For each modality i, attends to concatenated keys/values from all modalities.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: [F1, F2, F3], each (B, C, H, W) with same shapes
        if len(feats) != 3:
            raise ValueError("PA3WayBlock expects exactly three feature maps")
        b, c, h, w = feats[0].shape
        q = [self.q_proj(f).flatten(2).transpose(1, 2) for f in feats]  # (B, HW, C)
        k = [self.k_proj(f).flatten(2).transpose(1, 2) for f in feats]  # (B, HW, C)
        v = [self.v_proj(f).flatten(2).transpose(1, 2) for f in feats]  # (B, HW, C)

        k_all = torch.cat(k, dim=1)  # (B, 3*HW, C)
        v_all = torch.cat(v, dim=1)  # (B, 3*HW, C)

        fused = []
        scale = 1.0 / math.sqrt(c)
        for qi, fi in zip(q, feats):
            attn = torch.softmax(torch.bmm(qi, k_all.transpose(1, 2)) * scale, dim=-1)  # (B, HW, 3*HW)
            fused_tokens = torch.bmm(attn, v_all)  # (B, HW, C)
            fused_map = fused_tokens.transpose(1, 2).reshape(b, c, h, w)
            fused_map = self.out_proj(fused_map)
            fused_map = self.norm(fused_map)
            fused.append(fi + fused_map)  # residual fusion
        return fused


class MultiscalePAFusion(nn.Module):
    """
    Applies PA3WayBlock across multiple scales with shared channel sizes per scale.
    scales: list of feature keys in order, e.g., ["c1", "c2", "c3", "c4"].
    channels_per_scale: list of channel counts matching scales.
    """

    def __init__(self, scales: List[str], channels_per_scale: List[int]):
        super().__init__()
        assert len(scales) == len(channels_per_scale), "scales and channels_per_scale must align"
        self.scales = scales
        self.blocks = nn.ModuleDict(
            {s: PA3WayBlock(ch) for s, ch in zip(scales, channels_per_scale)}
        )

    def forward(self, feats1: dict, feats2: dict, feats3: dict) -> Tuple[dict, dict, dict]:
        out1, out2, out3 = {}, {}, {}
        for s in self.scales:
            f1, f2, f3 = feats1[s], feats2[s], feats3[s]
            fused = self.blocks[s]([f1, f2, f3])
            out1[s], out2[s], out3[s] = fused
        return out1, out2, out3


class ViTFuser(nn.Module):
    """
    Late fusion over tokens from all three modalities using a small Transformer encoder.
    """

    def __init__(self, in_channels: int, num_heads: int = 4, num_layers: int = 2, mlp_ratio: float = 4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=int(in_channels * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, C)
        fused = self.encoder(tokens)
        return fused.mean(dim=1)  # mean pooling over tokens


class ThreeModalFusionModel(nn.Module):
    """
    Three-branch CNN/ViT + multi-scale PA + ViT fusion head.
    Designed for 3 input modalities: (bmode, enhanced, improved).
    Supports multiple backbones: ResNet18, MAE ViT, DINO, i-JEPA.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        shared_weights: bool = True,
        classifier_dim: int = 256,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        # Build encoders
        shared = None
        if shared_weights:
            shared_encoder = create_multiscale_backbone(self.backbone_name, shared_backbone=None, in_channels=in_channels)
            if hasattr(shared_encoder, 'backbone'):
                shared = shared_encoder.backbone
            else:
                shared = shared_encoder
        
        self.enc1 = create_multiscale_backbone(self.backbone_name, shared_backbone=shared, in_channels=in_channels)
        self.enc2 = create_multiscale_backbone(self.backbone_name, shared_backbone=shared, in_channels=in_channels)
        self.enc3 = create_multiscale_backbone(self.backbone_name, shared_backbone=shared, in_channels=in_channels)

        # Determine channels based on backbone
        if self.backbone_name == "resnet18":
            self.scales = ["c1", "c2", "c3", "c4"]
            self.channels = [64, 128, 256, 512]
        elif self.backbone_name in ["mae_vit", "dino_vits8", "ijepa_vit"]:
            self.scales = ["c1", "c2", "c3", "c4"]
            # For ViT models, all scales have the same channel dimension (embed_dim)
            # Get embed_dim from first encoder
            if hasattr(self.enc1, 'embed_dim'):
                embed_dim = self.enc1.embed_dim
            else:
                # Fallback: try to infer from a forward pass
                dummy_input = torch.randn(1, in_channels, 224, 224)
                with torch.no_grad():
                    dummy_feats = self.enc1(dummy_input)
                    embed_dim = dummy_feats["c1"].shape[1]
            self.channels = [embed_dim] * 4  # All scales have same channels for ViT
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        self.fusion = MultiscalePAFusion(self.scales, self.channels)

        # Projection to a common dim for the ViT fuser
        self.proj = nn.Conv2d(self.channels[-1], classifier_dim, kernel_size=1)
        self.token_norm = nn.LayerNorm(classifier_dim)
        self.fuser = ViTFuser(in_channels=classifier_dim, num_heads=4, num_layers=2)
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        fused = self.extract_features(x1, x2, x3)
        logits = self.classifier(fused)
        return logits

    def extract_features(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        # Encode
        f1 = self.enc1(x1)
        f2 = self.enc2(x2)
        f3 = self.enc3(x3)

        # Multi-scale PA fusion
        f1, f2, f3 = self.fusion(f1, f2, f3)

        # Use highest-scale features for late fusion
        top1 = self.proj(f1[self.scales[-1]])
        top2 = self.proj(f2[self.scales[-1]])
        top3 = self.proj(f3[self.scales[-1]])

        # Flatten to tokens and stack
        def to_tokens(t: torch.Tensor) -> torch.Tensor:
            b, c, h, w = t.shape
            return t.flatten(2).transpose(1, 2)  # (B, HW, C)

        tokens = torch.cat([to_tokens(top1), to_tokens(top2), to_tokens(top3)], dim=1)
        tokens = self.token_norm(tokens)

        fused = self.fuser(tokens)
        return fused


def build_three_modal_resnet18(num_classes: int = 2, shared_weights: bool = True) -> nn.Module:
    """
    Helper to construct the three-modal fusion model with ResNet18 backbones.
    """
    return ThreeModalFusionModel(
        backbone_name="resnet18",
        num_classes=num_classes,
        shared_weights=shared_weights,
        classifier_dim=256,
    )


def build_three_modal_fusion(
    backbone_name: str = "resnet18",
    num_classes: int = 2,
    shared_weights: bool = True,
    classifier_dim: int = 256,
    in_channels: int = 3,
) -> nn.Module:
    """
    Generic helper to construct the three-modal fusion model with any supported backbone.
    
    Args:
        backbone_name: Name of the backbone ("resnet18", "mae_vit", "dino_vits8", "ijepa_vit")
        num_classes: Number of output classes
        shared_weights: Whether to share weights across the three encoders
        classifier_dim: Dimension of the final classifier features
        in_channels: Number of input channels (default: 3 for RGB)
    
    Returns:
        ThreeModalFusionModel instance
    """
    return ThreeModalFusionModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        shared_weights=shared_weights,
        classifier_dim=classifier_dim,
        in_channels=in_channels,
    )

