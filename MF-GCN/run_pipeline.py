import os
import re
import yaml
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import argparse
from pathlib import Path

from src.data import collect_split_dfs, make_loaders
from src.feature_extraction import train_cnn_and_save_best, extract_features_to_csv
from src.graphs import graph_cosine_topk, graph_sparse_random
from src.gcn_train import train_gcn, evaluate_gcn, save_graph_image
from src.cnn_backbones import get_backbone, requires_rgb, BACKBONES
from src.metrics import plot_confusion_matrix, save_metrics_csv
from src.fusion_mlp import train_mlp_and_save, export_mlp_features_csv, CustomFeatureDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def ensure_dir(path: str):
	if path and not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)


def wrap_rgb_transform(base_tf):
	if base_tf is None:
		return None
	return T.Compose([
		T.Grayscale(num_output_channels=3),
		base_tf,
	])


def resolved_run_id(cfg_run: dict) -> str:
	run_id = cfg_run.get("id")
	if not run_id or str(run_id).strip().lower() in {"none", "null", ""}:
		run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	return str(run_id)


def save_config_copy(cfg: dict, run_id: str, out_dir: str):
	ensure_dir(out_dir)
	with open(os.path.join(out_dir, f"config_{run_id}.yaml"), "w") as f:
		yaml.safe_dump(cfg, f)


def detect_available_modalities(base_dir: str, requested_modalities: list) -> list:
	"""
	Detect which modalities actually exist in the data directory.
	Returns list of modalities that have their own subdirectories.
	If a modality doesn't have its own directory, it will fall back to bmode images.
	"""
	available = []
	for modality in requested_modalities:
		modality_path = os.path.join(base_dir, "train", modality, "benign")
		simple_path = os.path.join(base_dir, "train", "benign")
		
		if os.path.isdir(modality_path):
			# Modality has its own directory
			available.append(modality)
		elif os.path.isdir(simple_path):
			# Fallback structure exists - modality will use same images as bmode
			if modality == "bmode":
				available.append(modality)
			# For non-bmode, we'll still process it but with a warning (handled in collect_split_dfs)
			else:
				available.append(modality)
	return available


def get_backbone_display_name(backbone: str) -> str:
	"""Convert backbone name to display name for run_id."""
	name_map = {
		"custom": "CustomCNN",
		"resnet18": "ResNet18",
		"dino_vits8": "DINO",
		"mae_vit": "MAE",
		"ijepa_vit": "IJEPA"
	}
	return name_map.get(backbone.lower(), backbone.upper())

def process_single_backbone(backbone: str, cfg: dict, device: torch.device, args=None):
	"""Process a single backbone: train CNN, extract features, concatenate, train GCN."""
	
	print(f"\n{'='*80}")
	print(f"Processing backbone: {backbone.upper()}")
	print(f"{'='*80}")
	
	# Create run_id based on backbone (without timestamp)
	run_cfg = cfg.get("run", {})
	backbone_display = get_backbone_display_name(backbone)
	# Use only backbone name, or custom ID if provided (no timestamp)
	run_id = run_cfg.get("id") if run_cfg.get("id") else backbone_display
	# Remove None/null values
	if not run_id or str(run_id).strip().lower() in {"none", "null", ""}:
		run_id = backbone_display
	
	stages = [s.lower() for s in run_cfg.get("stages", ["cnn", "gcn"])]
	prev_id = run_cfg.get("previous_run_id")

	base_dir = cfg["base_data_dir"]
	modalities = cfg["modalities"]
	
	# Check which modalities actually have their own directories
	modalities_with_dirs = []
	modalities_using_fallback = []
	for mod in modalities:
		mod_path = os.path.join(base_dir, "train", mod, "benign")
		if os.path.isdir(mod_path):
			modalities_with_dirs.append(mod)
		else:
			modalities_using_fallback.append(mod)
	
	# Print warning if any modalities are using fallback (same images as bmode)
	if modalities_using_fallback:
		print(f"\n⚠️  WARNING: The following modalities don't have their own directories:")
		for mod in modalities_using_fallback:
			print(f"   - {mod}: Will use same images as bmode (from {os.path.join(base_dir, 'train', 'benign')})")
		print(f"   This means you're processing the SAME images multiple times, not true multi-modal data!")
		print(f"   For true multi-modal analysis, create directories like:")
		for mod in modalities_using_fallback:
			print(f"     {os.path.join(base_dir, 'train', mod, 'benign')}")
		print()
	
	image_size = int(cfg["image_size"])
	batch_size = int(cfg["batch_size"])
	num_workers = int(cfg["num_workers"])

	# Get output directory (command-line arg takes precedence over config)
	output_dir = args.output_dir if args and args.output_dir else cfg.get("output_dir", "artifacts")
	
	# Ensure output directory exists
	ensure_dir(output_dir)

	# Update CNN config with current backbone
	cnn_cfg = cfg["cnn"].copy()
	cnn_cfg["backbone"] = backbone
	graph_cfg = cfg["graph"]
	gcn_cfg = cfg["gcn"]
	exp_cfg = cfg.get("experiments", {})
	fusion_cfg = cfg.get("fusion", {"method": "concat"})
	mlp_cfg = cfg.get("mlp", {})
	fusion_method = fusion_cfg.get("method", "concat").lower()

	root = os.path.join(output_dir, run_id)
	cnn_root = os.path.join(root, "cnn")
	feat_root = os.path.join(root, "features")
	fusion_root = os.path.join(root, "fusion")
	mlp_root = os.path.join(root, "mlp")
	gcn_root = os.path.join(root, "gcn")
	plots_root = os.path.join(root, "plots")

	# Save config copy with current backbone
	cfg_copy = cfg.copy()
	cfg_copy["cnn"] = cnn_cfg
	save_config_copy(cfg_copy, run_id, root)

	per_modality_csv = {}
	per_modality_cnn_preds = {}
	per_modality_cnn_labels = {}
	if "cnn" in stages:
		for modality in modalities:
			channels = 1 if modality == "bmode" else 3
			print(f"\n=== [CNN] Modality: {modality} (channels={channels}) ===")
			df_train, df_val, df_test = collect_split_dfs(base_dir, modality)

			# MAE-specific env wiring (optional)
			if str(backbone).lower() == "mae_vit":
				mae_ckpt = str(cnn_cfg.get("mae_checkpoint", "") or "").strip()
				mae_norm = str(cnn_cfg.get("mae_norm", "no_norm") or "no_norm").strip().lower()
				if mae_ckpt:
					os.environ["MAE_CKPT"] = mae_ckpt
				os.environ["MAE_NORM"] = mae_norm
			
			# i-JEPA-specific env wiring (optional)
			if str(backbone).lower() == "ijepa_vit":
				ijepa_model = str(cnn_cfg.get("ijepa_model", "facebook/ijepa_vith14_1k") or "facebook/ijepa_vith14_1k").strip()
				os.environ["IJEPA_MODEL"] = ijepa_model

			_, base_tf, aug_tf = get_backbone(backbone, in_channels=max(3, channels) if requires_rgb(backbone) else channels, feature_dim=int(cnn_cfg["feature_dim"]), num_classes=2)
			if requires_rgb(backbone) and channels == 1:
				base_tf = wrap_rgb_transform(base_tf)
				aug_tf = wrap_rgb_transform(aug_tf)

			train_loader, val_loader, test_loader = make_loaders(
				df_train, df_val, df_test, batch_size, num_workers, image_size, channels,
				override_transform_base=base_tf, override_transform_augment=aug_tf
			)

			best_model_path = train_cnn_and_save_best(
				train_loader=train_loader,
				val_loader=val_loader,
				in_channels=max(3, channels) if requires_rgb(backbone) else channels,
				feature_dim=int(cnn_cfg["feature_dim"]),
				save_dir=os.path.join(cnn_root, modality),
				lr=float(cnn_cfg["lr"]),
				weight_decay=float(cnn_cfg["weight_decay"]),
				epochs=int(cnn_cfg["epochs"]),
				patience=int(cnn_cfg["patience"]),
				device=device,
				backbone=backbone,
				freeze_backbone=bool(cnn_cfg.get("freeze_backbone", False)),
				train_last_n_layers=int(cnn_cfg.get("train_last_n_layers", 0)),
				plot_path=os.path.join(plots_root, f"cnn_loss_{modality}.png"),
				early_stopping_metric=str(cnn_cfg.get("early_stopping_metric", "accuracy")),
			)

			# Free GPU cache before large inference step (feature extraction)
			try:
				if device.type == "cuda":
					torch.cuda.empty_cache()
			except Exception:
				pass

			train_csv = os.path.join(feat_root, f"{modality}_train.csv")
			val_csv = os.path.join(feat_root, f"{modality}_val.csv")
			test_csv = os.path.join(feat_root, f"{modality}_test.csv")
			ensure_dir(os.path.dirname(train_csv))

			_, _, _ = extract_features_to_csv(best_model_path, max(3, channels) if requires_rgb(backbone) else channels, int(cnn_cfg["feature_dim"]), train_loader, train_csv, device, backbone=backbone)
			_, _, _ = extract_features_to_csv(best_model_path, max(3, channels) if requires_rgb(backbone) else channels, int(cnn_cfg["feature_dim"]), val_loader, val_csv, device, backbone=backbone)
			csv_path, test_preds, test_labels = extract_features_to_csv(best_model_path, max(3, channels) if requires_rgb(backbone) else channels, int(cnn_cfg["feature_dim"]), test_loader, test_csv, device, backbone=backbone)
			per_modality_csv[modality] = {"train": train_csv, "val": val_csv, "test": test_csv}
			per_modality_cnn_preds[modality] = test_preds
			per_modality_cnn_labels[modality] = test_labels

			plot_confusion_matrix(test_labels, test_preds, os.path.join(plots_root, f"cnn_cm_{modality}.png"), title=f"CNN ({backbone}:{modality}) Test Acc={100*(test_preds==test_labels).mean():.2f}%")
			
			# Clean up GPU memory after each modality
			if device.type == "cuda":
				torch.cuda.empty_cache()
	else:
		pid = prev_id if prev_id else run_id
		for modality in modalities:
			per_modality_csv[modality] = {
				"train": os.path.join(output_dir, pid, "features", f"{modality}_train.csv"),
				"val": os.path.join(output_dir, pid, "features", f"{modality}_val.csv"),
				"test": os.path.join(output_dir, pid, "features", f"{modality}_test.csv"),
			}

	# 2) Fusion stage: Concatenate features (always), then optionally apply MLP
	def normalize_image_name(img_name: str) -> str:
		"""
		Normalize image name by removing modality suffixes.
		Handles cases where enhanced/improved scans have suffixes like '_enhanced' or '_improved'.
		Examples:
		- "scan_001_enhanced.png" -> "scan_001.png"
		- "scan_001_improved.jpg" -> "scan_001.jpg"
		- "scan_001_enhanced" -> "scan_001"
		- "scan_001.png" -> "scan_001.png" (no change)
		"""
		img_str = str(img_name)
		# Pattern 1: Remove _enhanced or _improved followed by file extension
		# e.g., "scan_001_enhanced.png" -> "scan_001.png"
		normalized = re.sub(r'[._](enhanced|improved)(\.[^.]+)$', r'\2', img_str, flags=re.IGNORECASE)
		# Pattern 2: If no change, remove _enhanced or _improved at the end (no extension)
		# e.g., "scan_001_enhanced" -> "scan_001"
		if normalized == img_str:
			normalized = re.sub(r'[._](enhanced|improved)$', '', img_str, flags=re.IGNORECASE)
		return normalized
	
	def load_and_suffix(path, suffix, normalize_names=True):
		"""
		Load CSV and add suffix to feature columns.
		Also creates a normalized Image name for matching across modalities.
		"""
		df = pd.read_csv(path)
		
		# Create normalized image name for matching (before adding suffix to columns)
		if normalize_names and "Image" in df.columns:
			df["BaseImage"] = df["Image"].apply(normalize_image_name)
		else:
			df["BaseImage"] = df["Image"] if "Image" in df.columns else None
		
		# Add suffix to all columns except Image, Label, and BaseImage
		cols_to_suffix = [c for c in df.columns if c not in ["Image", "Label", "BaseImage"]]
		rename_dict = {c: f"{c}{suffix}" for c in cols_to_suffix}
		df = df.rename(columns=rename_dict)
		
		# Rename Image back (it got suffixed, but we want to keep original Image name)
		if f"Image{suffix}" in df.columns:
			df = df.rename(columns={f"Image{suffix}": "Image"})
		
		return df

	# Always create concatenated features first
	merged = {}
	for split in ["train", "val", "test"]:
		b = load_and_suffix(per_modality_csv["bmode"][split], "_B", normalize_names=True)
		e = load_and_suffix(per_modality_csv["enhanced"][split], "_E", normalize_names=True)
		q = load_and_suffix(per_modality_csv["improved"][split], "_Q", normalize_names=True)
		
		print(f"\n  Merging {split} features:")
		print(f"    Bmode: {len(b)} rows, Enhanced: {len(e)} rows, Improved: {len(q)} rows")
		
		# Merge on BaseImage (normalized) and Label to handle naming inconsistencies
		# Use bmode's Image column as the canonical image name
		# Note: BaseImage and Label are merge keys, so they won't get suffixes
		# Image columns will get suffixes, which we'll handle below
		m = b.merge(e, on=["BaseImage", "Label"], how="inner", suffixes=("", "_e"))
		print(f"    After bmode+enhanced merge: {len(m)} rows")
		
		m = m.merge(q, on=["BaseImage", "Label"], how="inner", suffixes=("", "_q"))
		print(f"    After adding improved: {len(m)} rows")
		
		# Remove the extra Image columns from enhanced and improved (keep bmode's Image)
		# pandas automatically adds suffixes to duplicate column names during merge
		cols_to_drop = []
		if "Image_e" in m.columns:
			cols_to_drop.append("Image_e")
		if "Image_q" in m.columns:
			cols_to_drop.append("Image_q")
		if cols_to_drop:
			m = m.drop(columns=cols_to_drop)
		
		# Remove BaseImage column (no longer needed after merge)
		if "BaseImage" in m.columns:
			m = m.drop(columns=["BaseImage"])
		
		b_cols = [c for c in m.columns if c.endswith("_B") and c not in ["Image", "Label"]]
		e_cols = [c for c in m.columns if c.endswith("_E") and c not in ["Image", "Label"]]
		q_cols = [c for c in m.columns if c.endswith("_Q") and c not in ["Image", "Label"]]
		m = m[["Image", "Label"] + b_cols + e_cols + q_cols]
		merged[split] = m
		print(f"    ✓ Final merged {split}: {len(m)} rows")

	ensure_dir(fusion_root)
	merged["train"].to_csv(os.path.join(fusion_root, "train_features_multi_modal.csv"), index=False)
	merged["val"].to_csv(os.path.join(fusion_root, "val_features_multi_modal.csv"), index=False)
	merged["test"].to_csv(os.path.join(fusion_root, "test_features_multi_modal.csv"), index=False)
	print(f"✓ Created concatenated multi-modal features")

	# Optionally train MLP and extract MLP features
	if fusion_method == "mlp":
		print(f"\n=== [MLP Fusion] Training MLP on concatenated features ===")
		ensure_dir(mlp_root)
		
		# Load concatenated features
		train_df = pd.read_csv(os.path.join(fusion_root, "train_features_multi_modal.csv"))
		val_df = pd.read_csv(os.path.join(fusion_root, "val_features_multi_modal.csv"))
		
		# Prepare data loaders for MLP
		def to_tensors(df):
			y = torch.tensor(df['Label'].to_numpy(), dtype=torch.long)
			feature_cols = [c for c in df.columns if c not in ['Image', 'Label']]
			x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
			img = df['Image'].to_numpy()
			return x, y, img
		
		x_train, y_train, img_train = to_tensors(train_df)
		x_val, y_val, img_val = to_tensors(val_df)
		
		train_loader_mlp = DataLoader(
			CustomFeatureDataset(x_train, y_train, img_train),
			batch_size=128, shuffle=True
		)
		val_loader_mlp = DataLoader(
			CustomFeatureDataset(x_val, y_val, img_val),
			batch_size=128, shuffle=False
		)
		
		input_dim = x_train.shape[1]
		hidden_dim1 = int(mlp_cfg.get("hidden_dim1", 512))
		hidden_dim2 = int(mlp_cfg.get("hidden_dim2", 256))
		
		# Train MLP
		best_mlp_path = train_mlp_and_save(
			train_loader=train_loader_mlp,
			val_loader=val_loader_mlp,
			input_dim=input_dim,
			hidden_dim1=hidden_dim1,
			hidden_dim2=hidden_dim2,
			lr=float(mlp_cfg.get("lr", 0.001)),
			momentum=float(mlp_cfg.get("momentum", 0.9)),
			weight_decay=float(mlp_cfg.get("weight_decay", 1.0e-4)),
			epochs=int(mlp_cfg.get("epochs", 50)),
			patience=int(mlp_cfg.get("patience", 10)),
			save_dir=mlp_root,
			device=device,
			plot_path=os.path.join(plots_root, "mlp_loss.png"),
			early_stopping_metric=str(mlp_cfg.get("early_stopping_metric", "accuracy")),
		)
		
		# Extract MLP features for all splits
		test_df = pd.read_csv(os.path.join(fusion_root, "test_features_multi_modal.csv"))
		x_test, y_test, img_test = to_tensors(test_df)
		test_loader_mlp = DataLoader(
			CustomFeatureDataset(x_test, y_test, img_test),
			batch_size=128, shuffle=False
		)
		
		mlp_train_csv = os.path.join(fusion_root, "train_features_mlp.csv")
		mlp_val_csv = os.path.join(fusion_root, "val_features_mlp.csv")
		mlp_test_csv = os.path.join(fusion_root, "test_features_mlp.csv")
		
		export_mlp_features_csv(
			best_mlp_path,
			(train_loader_mlp, val_loader_mlp, test_loader_mlp),
			(mlp_train_csv, mlp_val_csv, mlp_test_csv),
			device,
			input_dim=input_dim,
			hidden_dim1=hidden_dim1,
			hidden_dim2=hidden_dim2
		)
		print(f"✓ Extracted MLP features for GCN training")

	# 3) GCN stage - using MLP or concatenated features based on fusion method
	if "gcn" in stages:
		# Use MLP features if MLP fusion was used, otherwise use concatenated features
		if fusion_method == "mlp":
			train_df = pd.read_csv(os.path.join(fusion_root, "train_features_mlp.csv"))
			val_df = pd.read_csv(os.path.join(fusion_root, "val_features_mlp.csv"))
			test_df = pd.read_csv(os.path.join(fusion_root, "test_features_mlp.csv"))
			fusion_type_name = "MLP-fused"
		else:
			train_df = pd.read_csv(os.path.join(fusion_root, "train_features_multi_modal.csv"))
			val_df = pd.read_csv(os.path.join(fusion_root, "val_features_multi_modal.csv"))
			test_df = pd.read_csv(os.path.join(fusion_root, "test_features_multi_modal.csv"))
			fusion_type_name = "Concatenated"

		train_labels = train_df['Label'].values
		val_labels = val_df['Label'].values
		test_labels = test_df['Label'].values

		train_feats = train_df.drop(columns=['Label', 'Image']).values
		val_feats = val_df.drop(columns=['Label', 'Image']).values
		test_feats = test_df.drop(columns=['Label', 'Image']).values

		if graph_cfg["type"] == "cosine_topk":
			train_graph = graph_cosine_topk(train_feats, train_labels, train_df, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
			val_graph = graph_cosine_topk(val_feats, val_labels, val_df, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
			test_graph = graph_cosine_topk(test_feats, test_labels, test_df, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
		else:
			train_graph = graph_sparse_random(train_df, train_labels, max_neighbors=10)
			val_graph = graph_sparse_random(val_df, val_labels, max_neighbors=10)
			test_graph = graph_sparse_random(test_df, test_labels, max_neighbors=10)

		# Save graph images with homophily
		save_graph_image(train_graph, os.path.join(plots_root, f"graph_{fusion_method}_train.png"), title_prefix=f"{fusion_type_name} Multi-modal Train Graph")
		save_graph_image(val_graph, os.path.join(plots_root, f"graph_{fusion_method}_val.png"), title_prefix=f"{fusion_type_name} Multi-modal Val Graph")
		save_graph_image(test_graph, os.path.join(plots_root, f"graph_{fusion_method}_test.png"), title_prefix=f"{fusion_type_name} Multi-modal Test Graph")

		best_gcn = train_gcn(
			model_name=gcn_cfg["model"],
			train_graph=train_graph,
			val_graph=val_graph,
			hidden_channels=int(gcn_cfg["hidden_channels"]),
			lr=float(gcn_cfg["lr"]),
			weight_decay=float(gcn_cfg["weight_decay"]),
			epochs=int(gcn_cfg["epochs"]),
			patience=int(gcn_cfg["patience"]),
			aggregation_threshold=float(gcn_cfg["aggregation_threshold"]),
			save_dir=gcn_root,
			device=device,
			plot_path=os.path.join(plots_root, f"gcn_{fusion_method}_loss.png"),
			early_stopping_metric=str(gcn_cfg.get("early_stopping_metric", "accuracy")),
		)

		metrics = evaluate_gcn(
			model_name=gcn_cfg["model"],
			model_path=best_gcn,
			graph=test_graph,
			aggregation_threshold=float(gcn_cfg["aggregation_threshold"]),
			device=device,
		)
		print(f"GCN Test Metrics ({fusion_type_name} Multi-modal):", metrics)
		save_metrics_csv({**metrics, "modality": fusion_method}, os.path.join(root, f"gcn_metrics_{fusion_method}.csv"))

		from src.gcn_train import build_model
		model = build_model(gcn_cfg["model"], in_channels=test_graph.x.shape[1], hidden_channels=int(gcn_cfg["hidden_channels"]), num_classes=2, device=device)
		model.load_state_dict(torch.load(best_gcn, map_location=device))
		model.eval()
		with torch.no_grad():
			pred = model(test_graph.x.to(device), test_graph.edge_index.to(device)).argmax(1).cpu().numpy()
		
		# Save per-image predictions for multi-feature GCN
		predictions_df = pd.DataFrame({
			'Image': test_df['Image'].values,
			'Ground_Truth': test_labels,
			'Prediction': pred,
			'Correct': (pred == test_labels).astype(int)
		})
		predictions_df.to_csv(os.path.join(root, f"gcn_predictions_{fusion_method}.csv"), index=False)
		print(f"✓ Saved per-image predictions to gcn_predictions_{fusion_method}.csv")
		
		plot_confusion_matrix(test_labels, pred, os.path.join(plots_root, f"gcn_cm_{fusion_method}.png"), title=f"GCN ({fusion_type_name} Multi-modal) Test Acc={100*(pred==test_labels).mean():.2f}%")

		if exp_cfg.get("compare_bmode_vs_fusion", True):  # Default to True for comparison
			b_train = pd.read_csv(per_modality_csv["bmode"]["train"]).copy()
			b_val = pd.read_csv(per_modality_csv["bmode"]["val"]).copy()
			b_test = pd.read_csv(per_modality_csv["bmode"]["test"]).copy()
			b_train_labels = b_train['Label'].values
			b_val_labels = b_val['Label'].values
			b_test_labels = b_test['Label'].values
			b_train_feats = b_train.drop(columns=['Label', 'Image']).values
			b_val_feats = b_val.drop(columns=['Label', 'Image']).values
			b_test_feats = b_test.drop(columns=['Label', 'Image']).values

			if graph_cfg["type"] == "cosine_topk":
				b_train_graph = graph_cosine_topk(b_train_feats, b_train_labels, b_train, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
				b_val_graph = graph_cosine_topk(b_val_feats, b_val_labels, b_val, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
				b_test_graph = graph_cosine_topk(b_test_feats, b_test_labels, b_test, top_k=int(graph_cfg["top_k"]), normalize_zscore=True, normalize_l2=False)
			else:
				b_train_graph = graph_sparse_random(b_train, b_train_labels, max_neighbors=10)
				b_val_graph = graph_sparse_random(b_val, b_val_labels, max_neighbors=10)
				b_test_graph = graph_sparse_random(b_test, b_test_labels, max_neighbors=10)

			# Save graph images for bmode
			save_graph_image(b_train_graph, os.path.join(plots_root, "graph_bmode_train.png"), title_prefix="BMODE Train Graph")
			save_graph_image(b_val_graph, os.path.join(plots_root, "graph_bmode_val.png"), title_prefix="BMODE Val Graph")
			save_graph_image(b_test_graph, os.path.join(plots_root, "graph_bmode_test.png"), title_prefix="BMODE Test Graph")

			b_best_gcn = train_gcn(
				model_name=gcn_cfg["model"],
				train_graph=b_train_graph,
				val_graph=b_val_graph,
				hidden_channels=int(gcn_cfg["hidden_channels"]),
				lr=float(gcn_cfg["lr"]),
				weight_decay=float(gcn_cfg["weight_decay"]),
				epochs=int(gcn_cfg["epochs"]),
				patience=int(gcn_cfg["patience"]),
				aggregation_threshold=float(gcn_cfg["aggregation_threshold"]),
				save_dir=os.path.join(gcn_root, "bmode_only"),
				device=device,
				plot_path=os.path.join(plots_root, "gcn_bmode_loss.png"),
				early_stopping_metric=str(gcn_cfg.get("early_stopping_metric", "accuracy")),
			)

			b_metrics = evaluate_gcn(
				model_name=gcn_cfg["model"],
				model_path=b_best_gcn,
				graph=b_test_graph,
				aggregation_threshold=float(gcn_cfg["aggregation_threshold"]),
				device=device,
			)
			print("GCN Test Metrics (BMODE only):", b_metrics)
			save_metrics_csv({**b_metrics, "modality": "bmode"}, os.path.join(root, "gcn_metrics_bmode.csv"))

			model = build_model(gcn_cfg["model"], in_channels=b_test_graph.x.shape[1], hidden_channels=int(gcn_cfg["hidden_channels"]), num_classes=2, device=device)
			model.load_state_dict(torch.load(b_best_gcn, map_location=device))
			model.eval()
			with torch.no_grad():
				pred_b = model(b_test_graph.x.to(device), b_test_graph.edge_index.to(device)).argmax(1).cpu().numpy()
			
			# Save per-image predictions for bmode GCN
			b_predictions_df = pd.DataFrame({
				'Image': b_test['Image'].values,
				'Ground_Truth': b_test_labels,
				'Prediction': pred_b,
				'Correct': (pred_b == b_test_labels).astype(int)
			})
			b_predictions_df.to_csv(os.path.join(root, "gcn_predictions_bmode.csv"), index=False)
			print(f"✓ Saved per-image predictions to gcn_predictions_bmode.csv")
			
			plot_confusion_matrix(b_test_labels, pred_b, os.path.join(plots_root, "gcn_cm_bmode.png"), title=f"GCN (BMODE) Test Acc={100*(pred_b==b_test_labels).mean():.2f}%")
	
	# Clean up GPU memory before finishing
	if device.type == "cuda":
		torch.cuda.empty_cache()
	
	print(f"✓ Completed processing {backbone}")
	
	return run_id


def main():
	parser = argparse.ArgumentParser(description="Train all backbone models with concatenation fusion")
	parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
	parser.add_argument("--backbones", nargs="*", help="Specific backbones to train (default: all)", 
	                    choices=BACKBONES, default=None)
	parser.add_argument("--folds", nargs="*", help="Specific folds to run (e.g., fold1 fold2)", default=None)
	parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto-detect)")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default: 'artifacts' or from config)")
	
	args = parser.parse_args()
	
	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
	print(f"Using device: {device}")

	# Determine which backbones to process
	backbones_to_process = args.backbones if args.backbones else BACKBONES
	print(f"\n{'='*80}")
	print(f"Will train {len(backbones_to_process)} backbone(s): {backbones_to_process}")
	print(f"{'='*80}\n")

	# Get folds from config (default to processing all folds found or single directory)
	folds = cfg.get("folds", [])
	base_data_dir = cfg.get("base_data_dir", "")
	
	# Filter folds if specific ones were requested via command line
	if args.folds:
		if folds:
			# Filter the config folds to only include requested ones
			requested_folds = set(args.folds)
			folds = [f for f in folds if f in requested_folds]
			if not folds:
				print(f"Warning: None of the requested folds {args.folds} were found in config. Available folds: {cfg.get('folds', [])}")
			else:
				print(f"Command-line filter: Will process only specified folds: {folds}")
		else:
			# No folds in config but user specified folds
			folds = args.folds
			print(f"Will process folds from command line: {folds}")
	
	# If folds are specified and base_data_dir points to parent directory, process each fold
	# Otherwise, process as single directory
	if folds and base_data_dir:
		# Check if base_data_dir is a parent directory (doesn't end with fold name)
		base_path = Path(base_data_dir)
		# If the last part of the path starts with "fold", it's likely a specific fold directory
		process_folds = not base_path.name.lower().startswith("fold")
		
		if process_folds:
			print(f"Found {len(folds)} fold(s) to process: {folds}")
			print(f"Will process each fold separately for each backbone\n")
		else:
			folds = []  # Single directory, not folds
			print(f"Processing single directory: {base_data_dir}\n")
	
	# Process each backbone
	completed_runs = []
	for backbone in backbones_to_process:
		if folds:
			# Process each fold for this backbone
			for fold_name in folds:
				try:
					# Create a modified config for this fold
					fold_cfg = cfg.copy()
					fold_cfg["base_data_dir"] = os.path.join(base_data_dir, fold_name)
					fold_cfg["run"] = cfg.get("run", {}).copy()
					# Append fold name to run_id
					fold_cfg["run"]["id"] = f"{backbone}_{fold_name}" if not fold_cfg["run"].get("id") else f"{fold_cfg['run']['id']}_{fold_name}"
					
					print(f"\n{'='*80}")
					print(f"Processing {backbone.upper()} - {fold_name.upper()}")
					print(f"{'='*80}")
					
					run_id = process_single_backbone(backbone, fold_cfg, device, args)
					completed_runs.append((backbone, fold_name, run_id))
					
					# Clean GPU memory after each fold
					print(f"\n🧹 Cleaning GPU memory after {backbone} - {fold_name}...")
					if device.type == "cuda":
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
						# Force garbage collection
						import gc
						gc.collect()
						# Get memory stats
						try:
							allocated = torch.cuda.memory_allocated(device) / 1024**3
							reserved = torch.cuda.memory_reserved(device) / 1024**3
							print(f"   GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
						except:
							pass
					
				except Exception as e:
					print(f"\n❌ Error processing {backbone} - {fold_name}: {e}")
					import traceback
					traceback.print_exc()
					# Clean GPU memory even on error
					if device.type == "cuda":
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
						import gc
						gc.collect()
					continue
		else:
			# Process single directory (no folds)
			try:
				run_id = process_single_backbone(backbone, cfg, device, args)
				completed_runs.append((backbone, None, run_id))
			except Exception as e:
				print(f"\n❌ Error processing {backbone}: {e}")
				import traceback
				traceback.print_exc()
				continue

	print(f"\n{'='*80}")
	print("Training completed!")
	print(f"Processed {len(completed_runs)} run(s):")
	for run_info in completed_runs:
		if len(run_info) == 3 and run_info[1]:
			backbone, fold, run_id = run_info
			print(f"  - {backbone} ({fold}): {run_id}")
		else:
			backbone, _, run_id = run_info
			print(f"  - {backbone}: {run_id}")
	print(f"{'='*80}")


if __name__ == "__main__":
	main()
