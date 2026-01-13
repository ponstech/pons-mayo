import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CustomFeatureDataset(Dataset):
	def __init__(self, features: torch.Tensor, labels: torch.Tensor, image_names):
		self.features = features
		self.labels = labels
		self.image_names = image_names

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return self.features[idx], self.labels[idx], self.image_names[idx]


class MLP(nn.Module):
	def __init__(self, input_dim=1536, hidden_dim1=512, hidden_dim2=256, output_dim=2):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim1)
		self.bn1 = nn.BatchNorm1d(hidden_dim1)
		self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
		self.bn2 = nn.BatchNorm1d(hidden_dim2)
		self.fc3 = nn.Linear(hidden_dim2, output_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.relu(self.bn1(self.fc1(x)))
		x = self.dropout(x)
		x = self.relu(self.bn2(self.fc2(x)))
		x = self.dropout(x)
		return self.fc3(x)

	def extract_features(self, x):
		x = self.relu(self.bn1(self.fc1(x)))
		x = self.dropout(x)
		x = self.fc2(x)
		return x


def train_mlp_and_save(train_loader: DataLoader, val_loader: DataLoader, input_dim: int, hidden_dim1: int, hidden_dim2: int, lr: float, momentum: float, weight_decay: float, epochs: int, patience: int, save_dir: str, device: torch.device, plot_path: Optional[str] = None) -> str:
	os.makedirs(save_dir, exist_ok=True)
	model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=2).to(device)
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

	best_val_acc = 0.0
	counter = 0
	best_path = os.path.join(save_dir, "best_mlp.pth")
	train_losses: list[float] = []
	val_losses: list[float] = []

	for epoch in range(epochs):
		model.train()
		epoch_loss, correct, total = 0.0, 0, 0
		progress = tqdm(train_loader, desc=f"MLP | Epoch {epoch+1}/{epochs}", leave=False)
		for x, y, _ in progress:
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			out = model(x)
			loss = criterion(out, y)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			pred = out.argmax(1)
			correct += (pred == y).sum().item()
			total += y.size(0)
			avg_loss = epoch_loss / max(1, (progress.n or 1))
			progress.set_postfix(loss=f"{avg_loss:.4f}")

		# val
		model.eval()
		v_correct, v_total, v_loss = 0, 0, 0.0
		with torch.no_grad():
			for x, y, _ in val_loader:
				x, y = x.to(device), y.to(device)
				out = model(x)
				loss = criterion(out, y)
				v_loss += loss.item()
				pred = out.argmax(1)
				v_correct += (pred == y).sum().item()
				v_total += y.size(0)

		train_acc = 100.0 * correct / total if total > 0 else 0.0
		avg_train_loss = epoch_loss / max(1, len(train_loader))
		train_losses.append(avg_train_loss)
		val_acc = 100.0 * v_correct / v_total if v_total > 0 else 0.0
		avg_val_loss = v_loss / max(1, len(val_loader))
		val_losses.append(avg_val_loss)
		print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			counter = 0
			torch.save(model.state_dict(), best_path)
			print(f"Saved best MLP with Val Acc: {best_val_acc:.2f}%")
		else:
			counter += 1
			if counter >= patience:
				print("Early stopping MLP training")
				break

	# Save loss plot if requested
	if plot_path:
		os.makedirs(os.path.dirname(plot_path), exist_ok=True)
		plt.figure(figsize=(6,4))
		plt.plot(train_losses, label="Train Loss")
		plt.plot(val_losses, label="Val Loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.title("MLP Train/Val Loss")
		plt.legend()
		plt.tight_layout()
		plt.savefig(plot_path, dpi=150)
		plt.close()

	return best_path


def export_mlp_features_csv(model_path: str, loaders: Tuple[DataLoader, DataLoader, DataLoader], csv_paths: Tuple[str, str, str], device: torch.device, input_dim: int, hidden_dim1: int, hidden_dim2: int):
	model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=2).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()

	preds_out = []
	labels_out = []
	for loader, out_path in zip(loaders, csv_paths):
		features_list, labels_list, names_list, probs_all = [], [], [], []
		with torch.no_grad():
			batch_preds = []
			for x, y, names in loader:
				x = x.to(device)
				feat = model.extract_features(x).cpu().numpy()
				logits = model(x).cpu()
				probs = torch.softmax(logits, dim=1).numpy()
				pred = logits.argmax(1).numpy()
				features_list.append(feat)
				labels_list.extend(y.numpy())
				names_list.extend(list(names))
				probs_all.append(probs)
				batch_preds.extend(list(pred))

		features = np.concatenate(features_list) if len(features_list) else np.zeros((0, 256))
		probs_concat = np.concatenate(probs_all) if len(probs_all) else np.zeros((0, 2))
		df = pd.DataFrame(features)
		df['Label'] = np.array(labels_list)
		df['Image'] = names_list
		for i in range(probs_concat.shape[1]):
			df[f'Pseudo_Prob_Class_{i}'] = probs_concat[:, i]
		os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
		df.to_csv(out_path, index=False)

		preds_out.append(np.array(batch_preds))
		labels_out.append(np.array(labels_list))

	return preds_out, labels_out
