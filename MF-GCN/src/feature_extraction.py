import os
import tempfile
import shutil
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .cnn_models import CustomCNN
from .cnn_backbones import get_backbone, apply_freezing
from .data import make_loaders


def _to_class_indices(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
	# Ensure LongTensor of shape [N]
	if not isinstance(labels, torch.Tensor):
		labels = torch.tensor(labels)
	if labels.ndim == 2 and labels.size(1) == 1:
		labels = labels.squeeze(1)
	elif labels.ndim == 2 and labels.size(1) == num_classes:
		labels = labels.argmax(dim=1)
	return labels.long()


def _ensure_nc(outputs: torch.Tensor) -> torch.Tensor:
	# Reduce any extra dims to (N, C) by averaging across them
	if outputs.dim() > 2:
		reduce_dims = tuple(range(2, outputs.dim()))
		outputs = outputs.mean(dim=reduce_dims)
	return outputs


def train_cnn_and_save_best(train_loader, val_loader, in_channels: int, feature_dim: int, save_dir: str, lr: float, weight_decay: float, epochs: int, patience: int, device: torch.device, backbone: str = "custom", freeze_backbone: bool = False, train_last_n_layers: int = 0, plot_path: Optional[str] = None, early_stopping_metric: str = "accuracy") -> str:
	os.makedirs(save_dir, exist_ok=True)
	# Build backbone and possibly override transforms (handled outside in run_pipeline for loaders)
	model, _, _ = get_backbone(backbone, in_channels=in_channels, feature_dim=feature_dim, num_classes=2)
	# Apply freezing policy
	apply_freezing(model, backbone, freeze_backbone=freeze_backbone, train_last_n_layers=train_last_n_layers)
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

	# Initialize best values based on metric type
	early_stopping_metric = early_stopping_metric.lower()
	if early_stopping_metric == "loss":
		best_val_metric = float('inf')  # For loss, lower is better
	else:
		best_val_metric = 0.0  # For accuracy, higher is better
		early_stopping_metric = "accuracy"  # Default to accuracy
	
	counter = 0
	best_path = os.path.join(save_dir, f"best_{backbone}_in{in_channels}.pth")
	train_losses: list[float] = []
	val_losses: list[float] = []

	for epoch in range(epochs):
		# Train
		model.train()
		total, correct, total_loss = 0, 0, 0.0
		progress = tqdm(train_loader, desc=f"CNN {backbone}:{in_channels}ch | Epoch {epoch+1}/{epochs}", leave=False)
		for images, labels, _ in progress:
			try:
				images = images.to(device)
				labels = _to_class_indices(labels, num_classes=2).to(device=device, dtype=torch.long)

				optimizer.zero_grad()
				outputs = model(images)
				outputs = _ensure_nc(outputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
				pred = outputs.argmax(1)
				correct += (pred == labels).sum().item()
				total += labels.size(0)
				avg_loss = total_loss / max(1, (progress.n or 1))
				progress.set_postfix(loss=f"{avg_loss:.4f}")
				
				# Clear intermediate tensors
				del outputs, pred, loss
				if device.type == 'cuda':
					torch.cuda.empty_cache()
					
			except RuntimeError as e:
				if "out of memory" in str(e).lower():
					print(f"CUDA OOM during training: {e}")
					if device.type == 'cuda':
						torch.cuda.empty_cache()
					continue
				else:
					raise e

		train_acc = 100.0 * correct / total if total > 0 else 0.0
		avg_train_loss = total_loss / max(1, len(train_loader))
		train_losses.append(avg_train_loss)

		# Val
		model.eval()
		v_total, v_correct, v_loss = 0, 0, 0.0
		with torch.no_grad():
			for images, labels, _ in val_loader:
				try:
					images = images.to(device)
					labels = _to_class_indices(labels, num_classes=2).to(device=device, dtype=torch.long)
					outputs = model(images)
					outputs = _ensure_nc(outputs)
					loss = criterion(outputs, labels)
					v_loss += loss.item()
					pred = outputs.argmax(1)
					v_correct += (pred == labels).sum().item()
					v_total += labels.size(0)
					
					# Clear intermediate tensors
					del outputs, pred, loss
					if device.type == 'cuda':
						torch.cuda.empty_cache()
						
				except RuntimeError as e:
					if "out of memory" in str(e).lower():
						print(f"CUDA OOM during validation: {e}")
						if device.type == 'cuda':
							torch.cuda.empty_cache()
						continue
					else:
						raise e

		val_acc = 100.0 * v_correct / v_total if v_total > 0 else 0.0
		avg_val_loss = v_loss / max(1, len(val_loader))
		val_losses.append(avg_val_loss)

		print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

		# Determine current metric value and whether it's improved
		if early_stopping_metric == "loss":
			current_metric = avg_val_loss
			is_better = current_metric < best_val_metric
		else:  # accuracy
			current_metric = val_acc
			is_better = current_metric > best_val_metric
		
		if is_better:
			best_val_metric = current_metric
			counter = 0
			# Robust model saving for large models (especially HuggingFace models)
			try:
				# Try to save with legacy format first (more stable for large models)
				torch.save(model.state_dict(), best_path, _use_new_zipfile_serialization=False)
				if early_stopping_metric == "loss":
					print(f"Saved best CNN ({backbone}:{in_channels}ch) with Val Loss: {best_val_metric:.4f}")
				else:
					print(f"Saved best CNN ({backbone}:{in_channels}ch) with Val Acc: {best_val_metric:.2f}%")
			except Exception as e:
				print(f"Warning: Standard save failed, trying alternative method: {e}")
				try:
					# Alternative: save to temporary file first, then move
					import tempfile
					import shutil
					
					# Create temp file in same directory
					dirname = os.path.dirname(best_path) or "."
					os.makedirs(dirname, exist_ok=True)
					tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp_model_", suffix=".pth", dir=dirname)
					os.close(tmp_fd)
					
					# Save to temp file
					torch.save(model.state_dict(), tmp_path, _use_new_zipfile_serialization=False)
					
					# Atomic move
					shutil.move(tmp_path, best_path)
					if early_stopping_metric == "loss":
						print(f"Saved best CNN ({backbone}:{in_channels}ch) with Val Loss: {best_val_metric:.4f} (alternative method)")
					else:
						print(f"Saved best CNN ({backbone}:{in_channels}ch) with Val Acc: {best_val_metric:.2f}% (alternative method)")
				except Exception as e2:
					print(f"Error: Failed to save model: {e2}")
					print("Continuing training without saving...")
		else:
			counter += 1
			if counter >= patience:
				print(f"Early stopping CNN training (no improvement in val {early_stopping_metric} for {patience} epochs)")
				break

	# Save loss plot if requested
	if plot_path:
		os.makedirs(os.path.dirname(plot_path), exist_ok=True)
		plt.figure(figsize=(6,4))
		plt.plot(train_losses, label="Train Loss")
		plt.plot(val_losses, label="Val Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.title("CNN Train/Val Loss")
		plt.legend()
		plt.tight_layout()
		plt.savefig(plot_path, dpi=150)
		plt.close()

	# Clean up model from GPU memory (move to CPU first to release GPU memory)
	if device.type == 'cuda':
		model = model.cpu()
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	del model
	
	return best_path


def extract_features_to_csv(model_path: str, in_channels: int, feature_dim: int, data_loader, csv_path: str, device: torch.device, backbone: str = "custom"):
	model, _, _ = get_backbone(backbone, in_channels=in_channels, feature_dim=feature_dim, num_classes=2)
	model = model.to(device)
	
	# Robust model loading for large models with memory management
	try:
		# Clear GPU cache before loading
		if device.type == 'cuda':
			torch.cuda.empty_cache()
		
		# Try loading to CPU first to avoid GPU memory issues
		state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
		model.load_state_dict(state_dict)
		model = model.to(device)
		
	except Exception as e:
		print(f"Warning: Standard model loading failed: {e}")
		try:
			# Try loading with different approach
			if device.type == 'cuda':
				torch.cuda.empty_cache()
			state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
			model.load_state_dict(state_dict)
			model = model.to(device)
		except Exception as e2:
			raise RuntimeError(f"Failed to load model from {model_path}: {e2}")
	
	model.eval()

	all_features, all_labels, all_image_names, all_preds = [], [], [], []
	with torch.no_grad():
		for images, labels, image_names in tqdm(data_loader, desc="Extracting Features", leave=False):
			try:
				images = images.to(device)
				features = None
				if hasattr(model, 'extract_features'):
					features = model.extract_features(images).cpu().numpy()
				else:
					logits = model(images)
					features = logits.detach().cpu().numpy()
				preds = model(images).argmax(1).cpu().numpy()
				all_features.append(features)
				if isinstance(labels, torch.Tensor):
					labels = _to_class_indices(labels, num_classes=2).cpu().numpy()
				all_labels.extend(list(labels))
				all_image_names.extend(list(image_names))
				all_preds.extend(list(preds))
				
				# Clear intermediate tensors from GPU
				del features, preds
				if device.type == 'cuda':
					torch.cuda.empty_cache()
					
			except RuntimeError as e:
				if "out of memory" in str(e).lower():
					print(f"CUDA OOM during feature extraction: {e}")
					if device.type == 'cuda':
						torch.cuda.empty_cache()
					# Try with smaller batch or skip this batch
					print("Skipping this batch due to memory constraints...")
					continue
				else:
					raise e

	all_features = np.vstack(all_features) if len(all_features) else np.zeros((0, feature_dim))
	labels_arr = np.array(all_labels)
	df = pd.DataFrame(all_features)
	df.insert(0, "Image", all_image_names)
	df["Label"] = labels_arr
	os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
	df.to_csv(csv_path, index=False)
	
	# Clean up model from GPU memory (move to CPU first to release GPU memory)
	if device.type == 'cuda':
		model = model.cpu()
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	del model
	
	return csv_path, np.array(all_preds), labels_arr
