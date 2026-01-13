import os
import torch
import torchvision.transforms as T
import torch.nn as nn

from .cnn_models import CustomCNN

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


BACKBONES = ["custom", "resnet18", "dino_vits8", "mae_vit", "ijepa_vit"]


def requires_rgb(name: str) -> bool:
	return name.lower() in {"resnet18", "dino_vits8", "mae_vit", "ijepa_vit"}


class ResNetBackbone(nn.Module):
	def __init__(self, base: nn.Module, feature_dim: int, num_classes: int):
		super().__init__()
		self.backbone = base
		num_feats = base.fc.in_features
		self.backbone.fc = nn.Identity()
		self.feature_proj = nn.Linear(num_feats, feature_dim)
		self.classifier = nn.Linear(feature_dim, num_classes)
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(0.5)

	def forward_features(self, x):
		x = self.backbone.conv1(x)
		x = self.backbone.bn1(x)
		x = self.backbone.relu(x)
		x = self.backbone.maxpool(x)
		x = self.backbone.layer1(x)
		x = self.backbone.layer2(x)
		x = self.backbone.layer3(x)
		x = self.backbone.layer4(x)
		x = self.backbone.avgpool(x)
		x = torch.flatten(x, 1)
		return x

	def extract_features(self, x):
		feat = self.forward_features(x)
		feat = self.feature_proj(feat)
		return feat

	def forward(self, x):
		feat = self.extract_features(x)
		feat = self.act(feat)
		feat = self.drop(feat)
		return self.classifier(feat)


class DINOBackbone(nn.Module):
	def __init__(self, base: nn.Module, feature_dim: int, num_classes: int):
		super().__init__()
		self.backbone = base
		head_in = base.num_features
		self.feature_proj = nn.Linear(head_in, feature_dim)
		self.classifier = nn.Linear(feature_dim, num_classes)
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(0.5)

	def forward_features(self, x):
		feat = self.backbone.forward_features(x)
		# Normalize to (B, C): prefer CLS token, else mean over tokens
		if isinstance(feat, dict):
			feat = feat.get('x', None) or feat.get('features', None)
		if feat is None:
			raise RuntimeError("DINO forward_features returned unsupported structure")
		if feat.dim() == 3:
			# (B, tokens, C). If model has cls token at index 0, take it
			feat = feat[:, 0, :] if feat.size(1) >= 1 else feat.mean(dim=1)
		elif feat.dim() > 2:
			# pool spatial dims
			reduce_dims = tuple(range(2, feat.dim()))
			feat = feat.mean(dim=reduce_dims)
		return feat

	def extract_features(self, x):
		feat = self.forward_features(x)
		feat = self.feature_proj(feat)
		return feat

	def forward(self, x):
		feat = self.extract_features(x)
		feat = self.act(feat)
		feat = self.drop(feat)
		return self.classifier(feat)


class MAEViTBackbone(nn.Module):
	def __init__(self, base: nn.Module, feature_dim: int, num_classes: int):
		super().__init__()
		self.backbone = base
		head_in = getattr(base, 'num_features', None)
		if head_in is None and hasattr(base, 'embed_dim'):
			head_in = base.embed_dim
		if head_in is None:
			raise RuntimeError("Unsupported ViT backbone: cannot determine feature dimension")
		self.feature_proj = nn.Linear(head_in, feature_dim)
		self.classifier = nn.Linear(feature_dim, num_classes)
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(0.5)

	def forward_features(self, x):
		feat = self.backbone.forward_features(x)
		if isinstance(feat, dict):
			feat = feat.get('x', None) or feat.get('features', None)
		if feat is None:
			raise RuntimeError("MAE ViT forward_features returned unsupported structure")
		if feat.dim() == 3:
			feat = feat[:, 0, :] if feat.size(1) >= 1 else feat.mean(dim=1)
		elif feat.dim() > 2:
			reduce_dims = tuple(range(2, feat.dim()))
			feat = feat.mean(dim=reduce_dims)
		return feat

	def extract_features(self, x):
		feat = self.forward_features(x)
		feat = self.feature_proj(feat)
		return feat

	def forward(self, x):
		feat = self.extract_features(x)
		feat = self.act(feat)
		feat = self.drop(feat)
		return self.classifier(feat)


class IJEPABackbone(nn.Module):
	def __init__(self, base: nn.Module, feature_dim: int, num_classes: int):
		super().__init__()
		self.backbone = base
		# i-JEPA models from HuggingFace typically have config.embed_dim or config.hidden_size
		head_in = getattr(base.config, 'embed_dim', None) or getattr(base.config, 'hidden_size', None)
		if head_in is None:
			raise RuntimeError("Unsupported i-JEPA backbone: cannot determine feature dimension from config")
		self.feature_proj = nn.Linear(head_in, feature_dim)
		self.classifier = nn.Linear(feature_dim, num_classes)
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(0.5)

	def forward_features(self, x):
		# i-JEPA models from HuggingFace return features in last_hidden_state
		outputs = self.backbone(x, output_hidden_states=True)
		feat = outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
		
		# Take the CLS token (first token) or mean pool if no CLS token
		if feat.size(1) > 0:
			feat = feat[:, 0, :]  # CLS token
		else:
			feat = feat.mean(dim=1)  # Mean pooling
		return feat

	def extract_features(self, x):
		feat = self.forward_features(x)
		feat = self.feature_proj(feat)
		return feat

	def forward(self, x):
		feat = self.extract_features(x)
		feat = self.act(feat)
		feat = self.drop(feat)
		return self.classifier(feat)


def get_backbone(name: str, in_channels: int, feature_dim: int, num_classes: int = 2):
	name = name.lower()
	if name == "custom":
		model = CustomCNN(in_channels=in_channels, num_classes=num_classes, feature_dim=feature_dim)
		transform_base = None
		transform_aug = None
		return model, transform_base, transform_aug

	if name == "resnet18":
		if tvm is None:
			raise ImportError("torchvision not available for resnet18 backbone")
		base = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
		if in_channels == 1:
			base.conv1.in_channels = 1
			with torch.no_grad():
				base.conv1.weight = nn.Parameter(base.conv1.weight.mean(dim=1, keepdim=True))
		model = ResNetBackbone(base, feature_dim=feature_dim, num_classes=num_classes)
		weights = tvm.ResNet18_Weights.DEFAULT
		if in_channels == 3:
			transform_base = weights.transforms()
			transform_aug = weights.transforms()
		else:
			m = weights.meta.get("mean", (0.485, 0.456, 0.406))
			s = weights.meta.get("std", (0.229, 0.224, 0.225))
			mean_scalar = float(sum(m) / len(m))
			std_scalar = float(sum(s) / len(s))
			transform_base = T.Compose([
				T.Resize((224, 224)),
				T.ToTensor(),
				T.Normalize(mean=(mean_scalar,), std=(std_scalar,)),
			])
			transform_aug = transform_base
		return model, transform_base, transform_aug

	if name == "dino_vits8":
		if timm is None:
			raise ImportError("timm not available for DINO backbone")
		base = timm.create_model('vit_small_patch8_224.dino', pretrained=True)
		model = DINOBackbone(base, feature_dim=feature_dim, num_classes=num_classes)
		transform_base = T.Compose([
			T.Resize((224, 224)),
			T.ToTensor(),
			T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
		])
		transform_aug = transform_base
		return model, transform_base, transform_aug

	if name == "mae_vit":
		if timm is None:
			raise ImportError("timm not available for MAE ViT backbone")
		base = timm.create_model('vit_base_patch16_224', pretrained=False)
		ckpt_path = os.environ.get("MAE_CKPT", "").strip()
		if ckpt_path:
			try:
				# PyTorch 2.6 defaults weights_only=True; we need objects for some MAE checkpoints
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

		model = MAEViTBackbone(base, feature_dim=feature_dim, num_classes=num_classes)
		norm_policy = (os.environ.get("MAE_NORM", "no_norm") or "no_norm").lower()
		if norm_policy == "with_norm":
			transform_base = T.Compose([
				T.Resize((224, 224)),
				T.ToTensor(),
				T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225)),
			])
			transform_aug = transform_base
		else:
			transform_base = T.Compose([
				T.Resize((224, 224)),
				T.ToTensor(),
			])
			transform_aug = transform_base
		return model, transform_base, transform_aug

	if name == "ijepa_vit":
		if AutoModel is None or AutoImageProcessor is None:
			raise ImportError("transformers not available for i-JEPA backbone")
		
		# Use the official i-JEPA model from HuggingFace
		model_name = os.environ.get("IJEPA_MODEL", "facebook/ijepa_vith14_1k")
		try:
			base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
			processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
		except Exception as e:
			raise RuntimeError(f"Failed to load i-JEPA model '{model_name}': {e}")

		model = IJEPABackbone(base, feature_dim=feature_dim, num_classes=num_classes)
		
		# Use the processor's transforms if available, otherwise use standard ImageNet normalization
		if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
			size = processor.image_processor.size.get('height', 224)
		else:
			size = 224
			
		transform_base = T.Compose([
			T.Resize((size, size)),
			T.ToTensor(),
			T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		])
		transform_aug = transform_base
		return model, transform_base, transform_aug

	raise ValueError(f"Unknown backbone: {BACKBONES}")


def apply_freezing(model: nn.Module, backbone: str, freeze_backbone: bool, train_last_n_layers: int | None):
	name = backbone.lower()
	if name == "custom":
		return

	def set_requires_grad(mod, flag: bool):
		for p in mod.parameters():
			p.requires_grad = flag

	# If freeze_backbone is False, fine-tune everything (no freezing)
	if not freeze_backbone:
		print(f"Fine-tuning entire {name} architecture (no freezing)")
		return

	# If freeze_backbone is True but train_last_n_layers is 0, only train the head
	if freeze_backbone and (not train_last_n_layers or train_last_n_layers <= 0):
		print(f"Freezing {name} backbone, training only custom head")
		set_requires_grad(model, False)
		if hasattr(model, 'feature_proj'):
			set_requires_grad(model.feature_proj, True)
		if hasattr(model, 'classifier'):
			set_requires_grad(model.classifier, True)
		return

	# If freeze_backbone is True and train_last_n_layers > 0, freeze most but train last N layers + head
	print(f"Freezing {name} backbone, training last {train_last_n_layers} layers + custom head")
	set_requires_grad(model, False)

	# Always train the custom head
	if hasattr(model, 'feature_proj'):
		set_requires_grad(model.feature_proj, True)
	if hasattr(model, 'classifier'):
		set_requires_grad(model.classifier, True)

	backbone_mod = getattr(model, 'backbone', None)
	if backbone_mod is None:
		return

	# Apply layer-specific unfreezing based on backbone type
	if name == "resnet18":
		if train_last_n_layers and train_last_n_layers > 0:
			groups = [getattr(backbone_mod, f"layer{i}") for i in [1, 2, 3, 4]]
			for g in groups[-train_last_n_layers:]:
				set_requires_grad(g, True)
				print(f"Unfreezing ResNet layer: {g}")
	
	elif name == "dino_vits8":
		if hasattr(backbone_mod, "blocks") and train_last_n_layers and train_last_n_layers > 0:
			blocks = list(backbone_mod.blocks)
			for i, blk in enumerate(blocks[-train_last_n_layers:]):
				set_requires_grad(blk, True)
				print(f"Unfreezing DINO transformer block: {len(blocks) - train_last_n_layers + i + 1}/{len(blocks)}")
	
	elif name == "mae_vit":
		if hasattr(backbone_mod, "blocks") and train_last_n_layers and train_last_n_layers > 0:
			blocks = list(backbone_mod.blocks)
			for i, blk in enumerate(blocks[-train_last_n_layers:]):
				set_requires_grad(blk, True)
				print(f"Unfreezing MAE transformer block: {len(blocks) - train_last_n_layers + i + 1}/{len(blocks)}")
	
	elif name == "ijepa_vit":
		# For HuggingFace i-JEPA models, look for encoder layers
		if hasattr(backbone_mod, "encoder") and hasattr(backbone_mod.encoder, "layer"):
			if train_last_n_layers and train_last_n_layers > 0:
				layers = list(backbone_mod.encoder.layer)
				for i, blk in enumerate(layers[-train_last_n_layers:]):
					set_requires_grad(blk, True)
					print(f"Unfreezing i-JEPA encoder layer: {len(layers) - train_last_n_layers + i + 1}/{len(layers)}")
		elif hasattr(backbone_mod, "blocks") and train_last_n_layers and train_last_n_layers > 0:
			# Fallback for timm-style models
			blocks = list(backbone_mod.blocks)
			for i, blk in enumerate(blocks[-train_last_n_layers:]):
				set_requires_grad(blk, True)
				print(f"Unfreezing i-JEPA transformer block: {len(blocks) - train_last_n_layers + i + 1}/{len(blocks)}")
