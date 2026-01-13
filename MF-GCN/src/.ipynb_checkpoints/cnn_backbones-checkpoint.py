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


BACKBONES = ["custom", "resnet18", "dino_vits8"]


def requires_rgb(name: str) -> bool:
	return name.lower() in {"resnet18", "dino_vits8"}


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

	raise ValueError(f"Unknown backbone: {BACKBONES}")


def apply_freezing(model: nn.Module, backbone: str, freeze_backbone: bool, train_last_n_layers: int | None):
	name = backbone.lower()
	if name == "custom":
		return

	def set_requires_grad(mod, flag: bool):
		for p in mod.parameters():
			p.requires_grad = flag

	if not freeze_backbone and (not train_last_n_layers or train_last_n_layers <= 0):
		return

	set_requires_grad(model, False)

	if hasattr(model, 'feature_proj'):
		set_requires_grad(model.feature_proj, True)
	if hasattr(model, 'classifier'):
		set_requires_grad(model.classifier, True)

	backbone_mod = getattr(model, 'backbone', None)
	if backbone_mod is None:
		return

	if name == "resnet18":
		if train_last_n_layers and train_last_n_layers > 0:
			groups = [getattr(backbone_mod, f"layer{i}") for i in [1, 2, 3, 4]]
			for g in groups[-train_last_n_layers:]:
				set_requires_grad(g, True)
	elif name == "dino_vits8":
		if hasattr(backbone_mod, "blocks") and train_last_n_layers and train_last_n_layers > 0:
			for blk in list(backbone_mod.blocks)[-train_last_n_layers:]:
				set_requires_grad(blk, True)
