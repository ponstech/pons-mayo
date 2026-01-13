import os
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class BreastUltrasoundDataset(Dataset):
	def __init__(self, df: pd.DataFrame, transform=None, channels: int = 1):
		self.df = df.reset_index(drop=True)
		self.transform = transform
		self.channels = channels

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		img_path = self.df.loc[idx, "image_path"]
		label = int(self.df.loc[idx, "label"])
		image_name = os.path.basename(img_path)

		image = Image.open(img_path)
		# Convert based on channels
		if self.channels == 1:
			image = image.convert("L")
		else:
			# For 3-channel modalities, ensure image is RGB
			image = image.convert("RGB")

		if self.transform:
			image = self.transform(image)

		return image, torch.tensor(label, dtype=torch.long), image_name


def default_transforms(image_size: int = 224):
	base = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
	])
	augment = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(degrees=10),
	])
	return base, augment


def collect_split_dfs(base_dir: str, modality: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not os.path.isdir(base_dir):
		raise FileNotFoundError(
			f"Base data directory not found: {base_dir}. Update 'base_data_dir' in config.yaml to point to your dataset root.\n"
			"Expected structure: <base_dir>/(train|val|test)/<modality>/(benign|malignant)"
		)

	def load_image_paths_and_labels(base_path, label_value):
		if not os.path.isdir(base_path):
			return []
		image_files = [f for f in os.listdir(base_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
		return [(os.path.join(base_path, f), label_value) for f in image_files]

	train_benign = load_image_paths_and_labels(os.path.join(base_dir, "train", modality, "benign"), 0)
	train_malign = load_image_paths_and_labels(os.path.join(base_dir, "train", modality, "malignant"), 1)
	val_benign = load_image_paths_and_labels(os.path.join(base_dir, "val", modality, "benign"), 0)
	val_malign = load_image_paths_and_labels(os.path.join(base_dir, "val", modality, "malignant"), 1)
	test_benign = load_image_paths_and_labels(os.path.join(base_dir, "test", modality, "benign"), 0)
	test_malign = load_image_paths_and_labels(os.path.join(base_dir, "test", modality, "malignant"), 1)

	train_df = pd.DataFrame(train_benign + train_malign, columns=["image_path", "label"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
	val_df = pd.DataFrame(val_benign + val_malign, columns=["image_path", "label"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
	test_df = pd.DataFrame(test_benign + test_malign, columns=["image_path", "label"]).sample(frac=1.0, random_state=42).reset_index(drop=True)

	if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
		raise ValueError(
			f"No images found for modality '{modality}' under base_dir '{base_dir}'.\n"
			"Verify directory structure and that files exist in train/val/test splits."
		)

	return train_df, val_df, test_df


def make_loaders(
	df_train: pd.DataFrame,
	df_val: pd.DataFrame,
	df_test: pd.DataFrame,
	batch_size: int,
	num_workers: int,
	image_size: int,
	channels: int,
	override_transform_base: Optional[transforms.Compose] = None,
	override_transform_augment: Optional[transforms.Compose] = None,
):
	base, augment = default_transforms(image_size)
	if override_transform_base is not None:
		base = override_transform_base
	if override_transform_augment is not None:
		augment = override_transform_augment

	train_ds = BreastUltrasoundDataset(df_train, transform=augment, channels=channels)
	val_ds = BreastUltrasoundDataset(df_val, transform=base, channels=channels)
	test_ds = BreastUltrasoundDataset(df_test, transform=base, channels=channels)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, test_loader
