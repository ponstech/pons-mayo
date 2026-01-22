#!/usr/bin/env python3
"""
Single-image inference script for MF-GCN using:
- DINO ViT-Small backbone (dino_vits8)
- Concatenation fusion (3 × 512-D → 1536-D)
- Trained GCN checkpoint

================================================================================
USAGE INSTRUCTIONS
================================================================================

COMMAND LINE USAGE:
-------------------
Basic usage (using default artifacts directory):
    python inference_single.py \\
        --bmode /path/to/bmode_image.png \\
        --enhanced /path/to/enhanced_image.png \\
        --improved /path/to/improved_image.png

With custom artifacts directory:
    python inference_single.py \\
        --bmode /path/to/bmode_image.png \\
        --enhanced /path/to/enhanced_image.png \\
        --improved /path/to/improved_image.png \\
        --artifacts_root /path/to/artifacts

Save results to file:
    python inference_single.py \\
        --bmode /path/to/bmode_image.png \\
        --enhanced /path/to/enhanced_image.png \\
        --improved /path/to/improved_image.png \\
        --output results.json

All options:
    python inference_single.py \\
        --bmode <bmode_image_path> \\
        --enhanced <enhanced_image_path> \\
        --improved <improved_image_path> \\
        [--artifacts_root <artifacts_directory>] \\
        [--top_k <number>] \\
        [--gcn_model <model_name>] \\
        [--device <cuda|cpu>] \\
        [--output <output_file>] \\
        [--quiet]

Arguments:
    --bmode          (required) Path to B-mode ultrasound image
    --enhanced       (required) Path to Enhanced ultrasound image
    --improved       (required) Path to Improved ultrasound image
    --artifacts_root (optional) Path to artifacts directory (default: script directory)
    --top_k          (optional) Number of nearest neighbors for graph (default: 7)
    --gcn_model      (optional) GCN model name (default: GCNNClassifier)
    --device         (optional) Device to use: 'cuda' or 'cpu' (default: auto-detect)
    --output         (optional) Save results to file (.json or .txt)
    --quiet          (optional) Suppress console output (useful for GUI integration)

PROGRAMMATIC USAGE (for GUI/API integration):
--------------------------------------------
    from inference_single import infer_single
    
    try:
        results = infer_single(
            bmode_path="/path/to/bmode.png",
            enhanced_path="/path/to/enhanced.png",
            improved_path="/path/to/improved.png",
            artifacts_root=None,  # Uses default
            top_k=7,
            device_str="cuda",  # or "cpu"
            output_file=None,   # Optional: save to file
            quiet=True          # Suppress console output
        )
        
        # Access results
        prediction = results['predicted_label']  # "benign" or "malignant"
        confidence = results['confidence']        # 0.0 to 1.0
        prob_benign = results['probabilities']['benign']
        prob_malignant = results['probabilities']['malignant']
        
    except Exception as e:
        # Handle errors appropriately
        print(f"Error: {e}")

RETURN VALUE STRUCTURE:
-----------------------
The function returns a dictionary with:
    {
        'predicted_class': int,           # 0=benign, 1=malignant
        'predicted_label': str,           # "benign" or "malignant"
        'logits': {
            'benign': float,
            'malignant': float
        },
        'probabilities': {
            'benign': float,              # 0.0 to 1.0
            'malignant': float            # 0.0 to 1.0
        },
        'confidence': float,              # Probability of predicted class
        'input_images': {
            'bmode': str,
            'enhanced': str,
            'improved': str
        },
        'model_info': {
            'backbone': str,
            'gcn_model': str,
            'gcn_checkpoint': str,
            'top_k_neighbors': int,
            'device': str,
            'artifacts_root': str
        },
        'timestamp': str                  # ISO format timestamp
    }

REQUIRED DIRECTORY STRUCTURE:
-----------------------------
The artifacts directory should contain:
    artifacts_root/
    ├── cnn/
    │   ├── bmode/
    │   │   └── best_dino_vits8_in3.pth
    │   ├── enhanced/
    │   │   └── best_dino_vits8_in3.pth
    │   └── improved/
    │       └── best_dino_vits8_in3.pth
    ├── fusion/
    │   └── train_features_multi_modal.csv
    └── gcn/
        └── best_GCNNClassifier.pth

IMAGE REQUIREMENTS:
-------------------
- All three images (B-mode, Enhanced, Improved) must be provided
- Images should be in common formats: PNG, JPEG, JPG, etc.
- Images will be automatically resized to 224x224 for DINO processing
- B-mode images will be converted to RGB (3-channel) for DINO compatibility

OUTPUT:
-------
- Console output: Results printed to terminal (unless --quiet is used)
- File output: If --output is specified, results saved to file
- Return value: Dictionary with all results (for programmatic use)

ERROR HANDLING:
---------------
The script validates:
- Image files exist and are readable
- Artifacts directory structure is correct
- Model checkpoints are available
- All required files are present

Errors are raised as exceptions with clear messages for debugging.

================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

# Add parent directory to path to import from src/
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from src.cnn_backbones import get_backbone, requires_rgb
from src.graphs import _normalize_features
from src.gcn_train import build_model


def load_dino_backbones(
    artifacts_root: Path,
    device: torch.device,
    feature_dim: int = 512,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, callable]:
    """
    Load three DINO feature extractors (bmode, enhanced, improved) and the base transform.

    Assumes that CNN training artifacts are stored under:
        artifacts_root / "cnn" / {modality} / best_dino_vits8_in3.pth
    where modality ∈ {"bmode", "enhanced", "improved"}.
    
    All DINO models use 3 input channels (RGB) even for B-mode images.
    """
    modalities = ["bmode", "enhanced", "improved"]
    channels_map = {"bmode": 1, "enhanced": 3, "improved": 3}

    models = {}
    base_tf = None

    for mod in modalities:
        in_ch = channels_map[mod]
        ckpt_path = artifacts_root / "cnn" / mod / f"best_dino_vits8_in{max(3, in_ch) if requires_rgb('dino_vits8') else in_ch}.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing CNN checkpoint for {mod}: {ckpt_path}")

        model, transform_base, _ = get_backbone(
            "dino_vits8",
            in_channels=max(3, in_ch) if requires_rgb("dino_vits8") else in_ch,
            feature_dim=feature_dim,
            num_classes=2,
        )
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)  # Use strict=False to handle potential mismatches
        model.to(device)
        model.eval()
        
        # Verify model is in eval mode and has extract_features method
        if not hasattr(model, "extract_features"):
            raise RuntimeError(f"Model for {mod} does not have extract_features method")
        
        models[mod] = model

        # Use the same base transform for all three
        if base_tf is None:
            base_tf = transform_base

    if base_tf is None:
        raise RuntimeError("Failed to obtain base transform for DINO backbone")

    return models["bmode"], models["enhanced"], models["improved"], base_tf


def load_training_features_concat(
    features_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training fused features (concatenation) from CSV.

    Expected columns: Image, Label, f0, f1, ..., f{D-1}
    """
    if not features_csv.is_file():
        raise FileNotFoundError(f"Training features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)
    if "Label" not in df.columns or "Image" not in df.columns:
        raise ValueError("Expected 'Image' and 'Label' columns in features CSV")

    labels = df["Label"].values.astype(int)
    image_names = df["Image"].values
    features = df.drop(columns=["Image", "Label"]).values.astype(np.float32)
    return features, labels, image_names


def extract_single_features_concat(
    bmode_path: Path,
    enhanced_path: Path,
    improved_path: Path,
    model_bmode: torch.nn.Module,
    model_enh: torch.nn.Module,
    model_imp: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    """
    Extract 512-D features for each feature type using DINO backbone and concatenate.
    """
    def _load_image(path: Path, rgb: bool = True) -> torch.Tensor:
        img = Image.open(path).convert("RGB" if rgb else "L")
        return transform(img)

    # All DINO models expect RGB (requires_rgb('dino_vits8') is True)
    b_img = _load_image(bmode_path, rgb=True).unsqueeze(0).to(device)
    e_img = _load_image(enhanced_path, rgb=True).unsqueeze(0).to(device)
    i_img = _load_image(improved_path, rgb=True).unsqueeze(0).to(device)

    with torch.no_grad():
        # Ensure models are in eval mode
        model_bmode.eval()
        model_enh.eval()
        model_imp.eval()
        
        # Extract features
        if hasattr(model_bmode, "extract_features"):
            fb = model_bmode.extract_features(b_img)
        else:
            # Fallback: use forward and take features before classifier
            fb = model_bmode(b_img)
            # If it's logits, we need to get features differently
            # This shouldn't happen with DINOBackbone, but handle it
            if fb.shape[-1] == 2:  # Looks like logits
                raise RuntimeError("Model returned logits instead of features. Check model architecture.")
        
        if hasattr(model_enh, "extract_features"):
            fe = model_enh.extract_features(e_img)
        else:
            fe = model_enh(e_img)
            if fe.shape[-1] == 2:
                raise RuntimeError("Model returned logits instead of features. Check model architecture.")
        
        if hasattr(model_imp, "extract_features"):
            fi = model_imp.extract_features(i_img)
        else:
            fi = model_imp(i_img)
            if fi.shape[-1] == 2:
                raise RuntimeError("Model returned logits instead of features. Check model architecture.")

    # Ensure we have the right shape: (batch, features) -> (features,)
    fb = fb.squeeze(0).cpu().numpy().astype(np.float32)
    fe = fe.squeeze(0).cpu().numpy().astype(np.float32)
    fi = fi.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Verify feature dimensions
    if len(fb.shape) != 1 or len(fe.shape) != 1 or len(fi.shape) != 1:
        raise ValueError(f"Expected 1D features, got shapes: bmode={fb.shape}, enhanced={fe.shape}, improved={fi.shape}")
    
    if fb.shape[0] != 512 or fe.shape[0] != 512 or fi.shape[0] != 512:
        raise ValueError(f"Expected 512-D features, got: bmode={fb.shape[0]}, enhanced={fe.shape[0]}, improved={fi.shape[0]}")

    fused = np.concatenate([fb, fe, fi], axis=0)
    return fused  # shape (1536,)


def build_inference_graph(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    new_feat: np.ndarray,
    top_k: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a small graph consisting of:
    - N training nodes
    - 1 new node (index N)

    Edges:
    - Original training-training edges are NOT reconstructed (for simplicity).
    - New node connects to its top-K most similar training nodes (cosine similarity), undirected.
    """
    # Normalize features as in graph_cosine_topk
    # IMPORTANT: Normalize new_feat using the training data statistics, not its own!
    # This ensures consistent normalization across all samples
    all_feats = np.vstack([train_feats, new_feat[None, :]])
    all_norm = _normalize_features(all_feats, zscore=True, l2=False)
    train_norm = all_norm[:-1]  # All but last
    new_norm = all_norm[-1]     # Last one (the new sample)

    # Compute cosine similarity between new node and all training nodes
    norms_train = np.linalg.norm(train_norm, axis=1, keepdims=True)
    norms_train[norms_train == 0] = 1e-9
    norm_new = np.linalg.norm(new_norm) + 1e-9
    cos_sim = (train_norm @ new_norm) / (norms_train.squeeze() * norm_new)  # (N,)

    # Select top-K neighbors
    k = min(top_k, train_norm.shape[0])
    neighbor_idx = np.argsort(cos_sim)[-k:]

    # Build feature matrix x: [train_norm; new_norm]
    x_all = np.vstack([train_norm, new_norm[None, :]]).astype(np.float32)
    x = torch.tensor(x_all, dtype=torch.float32)

    # Build labels: keep training labels; dummy label for new node (not used in inference)
    y_all = np.concatenate([train_labels, np.array([-1], dtype=train_labels.dtype)])
    y = torch.tensor(y_all, dtype=torch.long)

    # Edge index: undirected edges between new node (idx = N) and neighbors
    N = train_norm.shape[0]
    edges = []
    for j in neighbor_idx:
        edges.append([N, int(j)])
        edges.append([int(j), N])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return x, edge_index, y


def validate_image_path(image_path: Path, image_type: str) -> None:
    """Validate that an image file exists and is readable."""
    if not image_path.exists():
        raise FileNotFoundError(f"{image_type} image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"{image_type} path is not a file: {image_path}")
    try:
        # Try to open and verify it's a valid image
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"{image_type} image is not a valid image file ({image_path}): {e}")


def infer_single(
    bmode_path: str,
    enhanced_path: str,
    improved_path: str,
    artifacts_root: str = None,
    backbone: str = "dino_vits8",
    gcn_model_name: str = "GCNNClassifier",
    top_k: int = 7,
    device_str: str = None,
    output_file: Optional[str] = None,
    quiet: bool = False,
) -> Dict:
    """
    High-level API to run single-sample inference with MF-GCN (DINO + concat).
    
    If artifacts_root is None, uses the dino_gcn_inference directory (where this script is located).
    
    Args:
        bmode_path: Path to B-mode ultrasound image
        enhanced_path: Path to Enhanced ultrasound image
        improved_path: Path to Improved ultrasound image
        artifacts_root: Path to artifacts directory (default: script directory)
        backbone: Backbone model name (default: "dino_vits8")
        gcn_model_name: GCN model name (default: "GCNNClassifier")
        top_k: Number of nearest neighbors for graph construction (default: 7)
        device_str: Device to use ("cuda" or "cpu", default: auto-detect)
        output_file: Optional path to save results (.json or .txt)
        quiet: If True, suppress console output (useful for GUI integration)
    
    Returns:
        Dictionary containing:
        - 'predicted_class': int (0=benign, 1=malignant)
        - 'predicted_label': str ("benign" or "malignant")
        - 'logits': dict with 'benign' and 'malignant' keys
        - 'probabilities': dict with 'benign' and 'malignant' keys
        - 'confidence': float (probability of predicted class)
        - 'input_images': dict with paths to input images
        - 'model_info': dict with model and configuration info
        - 'timestamp': str (ISO format timestamp)
    
    Raises:
        FileNotFoundError: If required files or images are missing
        ValueError: If images are invalid or parameters are incorrect
        RuntimeError: If model loading or inference fails
    """
    # Validate input images
    bmode_path_obj = Path(bmode_path)
    enhanced_path_obj = Path(enhanced_path)
    improved_path_obj = Path(improved_path)
    
    validate_image_path(bmode_path_obj, "B-mode")
    validate_image_path(enhanced_path_obj, "Enhanced")
    validate_image_path(improved_path_obj, "Improved")
    
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Default to the directory containing this script
    if artifacts_root is None:
        artifacts_root = Path(__file__).parent
    else:
        artifacts_root = Path(artifacts_root)
    
    # Features CSV: concatenated fused features from training set
    # Located in fusion/ subdirectory
    fusion_dir = artifacts_root / "fusion"
    train_feat_csv = fusion_dir / "train_features_multi_modal.csv"

    # Load training features and labels
    train_feats, train_labels, _ = load_training_features_concat(train_feat_csv)

    # Load DINO backbones and transforms
    model_b, model_e, model_i, base_tf = load_dino_backbones(artifacts_root, device=device)

    # Extract concatenated feature for the new sample
    new_feat = extract_single_features_concat(
        Path(bmode_path),
        Path(enhanced_path),
        Path(improved_path),
        model_b,
        model_e,
        model_i,
        base_tf,
        device=device,
    )

    # Build inference graph
    x, edge_index, y = build_inference_graph(train_feats, train_labels, new_feat, top_k=top_k)

    # Load trained GCN
    gcn_dir = artifacts_root / "gcn"
    # Look for best_GCNNClassifier.pth specifically, or fall back to any best_*.pth
    gcn_ckpt = gcn_dir / f"best_{gcn_model_name}.pth"
    if not gcn_ckpt.is_file():
        # Fallback: try to find any best_*.pth file
        ckpt_candidates = list(gcn_dir.glob("best_*.pth"))
        if not ckpt_candidates:
            raise FileNotFoundError(f"No GCN checkpoint found in {gcn_dir}. Expected: {gcn_ckpt}")
        gcn_ckpt = ckpt_candidates[0]
        if not quiet:
            print(f"Warning: Using {gcn_ckpt.name} instead of best_{gcn_model_name}.pth")

    in_channels = x.shape[1]
    gcn_model = build_model(
        name=gcn_model_name,
        in_channels=in_channels,
        hidden_channels=256,
        num_classes=2,
        device=device,
    )
    state_dict = torch.load(str(gcn_ckpt), map_location=device)
    gcn_model.load_state_dict(state_dict)
    gcn_model.to(device)
    gcn_model.eval()

    with torch.no_grad():
        out = gcn_model(x.to(device), edge_index.to(device))
        logits_new = out[-1]  # last node is the new sample
        probs_new = F.softmax(logits_new, dim=0)
        pred_class = int(logits_new.argmax().item())

    # Convert to numpy for output
    logits_arr = logits_new.cpu().numpy()
    probs_arr = probs_new.cpu().numpy()
    confidence = float(probs_arr[pred_class])

    # Prepare results dictionary
    results = {
        "predicted_class": pred_class,
        "predicted_label": "benign" if pred_class == 0 else "malignant",
        "logits": {
            "benign": float(logits_arr[0]),
            "malignant": float(logits_arr[1])
        },
        "probabilities": {
            "benign": float(probs_arr[0]),
            "malignant": float(probs_arr[1])
        },
        "confidence": confidence,
        "input_images": {
            "bmode": str(bmode_path),
            "enhanced": str(enhanced_path),
            "improved": str(improved_path)
        },
        "model_info": {
            "backbone": backbone,
            "gcn_model": gcn_model_name,
            "gcn_checkpoint": str(gcn_ckpt.name),
            "top_k_neighbors": top_k,
            "device": str(device),
            "artifacts_root": str(artifacts_root)
        },
        "timestamp": datetime.now().isoformat()
    }

    # Print to console (unless quiet mode)
    if not quiet:
        print("\n" + "="*60)
        print("Single-sample MF-GCN Inference (DINO + Concat)")
        print("="*60)
        print(f"Device: {device}")
        print(f"Artifacts root: {artifacts_root}")
        print(f"GCN checkpoint: {gcn_ckpt.name}")
        print(f"Top-K neighbors used: {top_k}")
        print("\n" + "-"*60)
        print("RESULTS:")
        print("-"*60)
        print(f"Predicted class: {pred_class} ({results['predicted_label']})")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"\nLogits:")
        print(f"  Benign:    {logits_arr[0]:.4f}")
        print(f"  Malignant: {logits_arr[1]:.4f}")
        print(f"\nProbabilities:")
        print(f"  Benign:    {probs_arr[0]:.4f} ({probs_arr[0]*100:.2f}%)")
        print(f"  Malignant: {probs_arr[1]:.4f} ({probs_arr[1]*100:.2f}%)")
        print("="*60)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            if not quiet:
                print(f"\nResults saved to: {output_path}")
        else:
            # Save as text file
            with open(output_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("MF-GCN Inference Results\n")
                f.write("="*60 + "\n\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                f.write("Input Images:\n")
                f.write(f"  B-mode:   {results['input_images']['bmode']}\n")
                f.write(f"  Enhanced: {results['input_images']['enhanced']}\n")
                f.write(f"  Improved: {results['input_images']['improved']}\n\n")
                f.write("Model Configuration:\n")
                f.write(f"  Backbone: {results['model_info']['backbone']}\n")
                f.write(f"  GCN Model: {results['model_info']['gcn_model']}\n")
                f.write(f"  Top-K: {results['model_info']['top_k_neighbors']}\n")
                f.write(f"  Device: {results['model_info']['device']}\n\n")
                f.write("-"*60 + "\n")
                f.write("PREDICTION:\n")
                f.write("-"*60 + "\n")
                f.write(f"Class: {results['predicted_class']} ({results['predicted_label']})\n")
                f.write(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)\n\n")
                f.write("Logits:\n")
                f.write(f"  Benign:    {results['logits']['benign']:.4f}\n")
                f.write(f"  Malignant: {results['logits']['malignant']:.4f}\n\n")
                f.write("Probabilities:\n")
                f.write(f"  Benign:    {results['probabilities']['benign']:.4f} ({results['probabilities']['benign']*100:.2f}%)\n")
                f.write(f"  Malignant: {results['probabilities']['malignant']:.4f} ({results['probabilities']['malignant']*100:.2f}%)\n")
                f.write("="*60 + "\n")
            if not quiet:
                print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Single-image MF-GCN inference (DINO + concat)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default artifacts directory (dino_gcn_inference/)
  python inference_single.py --bmode img_b.png --enhanced img_e.png --improved img_i.png
  
  # Specify custom artifacts directory
  python inference_single.py --bmode img_b.png --enhanced img_e.png --improved img_i.png \\
      --artifacts_root /path/to/artifacts
        """
    )
    parser.add_argument("--bmode", type=str, required=True, help="Path to B-mode image")
    parser.add_argument("--enhanced", type=str, required=True, help="Path to enhanced image")
    parser.add_argument("--improved", type=str, required=True, help="Path to improved image")
    parser.add_argument(
        "--artifacts_root",
        type=str,
        default=None,
        help="Path to artifacts directory (default: dino_gcn_inference/ directory containing this script)",
    )
    parser.add_argument("--top_k", type=int, default=7, help="Top-K neighbors for graph construction")
    parser.add_argument("--gcn_model", type=str, default="GCNNClassifier", help="GCN model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to file (supports .json or .txt format). Example: --output results.json"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful for GUI/API integration)"
    )

    args = parser.parse_args()

    try:
        results = infer_single(
            bmode_path=args.bmode,
            enhanced_path=args.enhanced,
            improved_path=args.improved,
            artifacts_root=args.artifacts_root,
            backbone="dino_vits8",
            gcn_model_name=args.gcn_model,
            top_k=args.top_k,
            device_str=args.device,
            output_file=args.output,
            quiet=args.quiet,
        )
        return results
    except Exception as e:
        if args.quiet:
            # In quiet mode, just raise the exception
            raise
        else:
            # In normal mode, print error and exit
            print(f"\n❌ Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()


