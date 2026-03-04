"""
Extract test embeddings and compute distances to keywords.

Outputs:
  - Embeddings (and labels) for the test set
  - Per-sample distances to each keyword (for comparison: B-mode only vs multi-feature)
"""

import argparse
import os
import json

from .warnings_config import setup_warnings
setup_warnings()

import torch
import numpy as np
from torch.utils.data import DataLoader

from .model import PONSCLIPModel
from .dataset import BreastUSDataset, BreastUSFolderDataset, collate_fn, get_default_image_transform
from .constants import KEYWORDS, BREAST_US_CLASSES


def get_processor():
    try:
        from medclip import MedCLIPProcessor
        return MedCLIPProcessor()
    except ImportError as e:
        raise ImportError(
            "PONS-CLIP requires MedCLIPProcessor from the medclip package for evaluation. "
            "Install with: pip install medclip (or pip install git+https://github.com/RyanWangZf/MedCLIP.git). "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to load MedCLIPProcessor. Evaluation cannot fall back to another tokenizer. "
            f"Original error: {e}"
        ) from e


def load_model(checkpoint_path: str, use_multifeature: bool, device: str, fusion_type: str = "weighted"):
    from medclip import MedCLIPVisionModelViT
    model = PONSCLIPModel(
        vision_cls=MedCLIPVisionModelViT,
        use_multifeature=use_multifeature,
        fusion_type=fusion_type,
    )
    model.from_pretrained()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model.to(device).eval()


def encode_keywords(model: PONSCLIPModel, processor, device: str) -> torch.Tensor:
    """Encode KEYWORDS to (num_keywords, dim) normalized embeddings."""
    # Support both HF tokenizer and MedCLIPProcessor-style
    if hasattr(processor, "tokenizer"):
        enc = processor.tokenizer(
            KEYWORDS,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
    else:
        enc = processor(
            KEYWORDS,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(input_ids, attention_mask)
    return text_embeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to PONS-CLIP checkpoint")
    # Option A: folder-based (data/fold1 with test/ under it)
    p.add_argument("--data_dir", type=str, default=None, help="e.g. data/fold1 — use test/ under it (overrides manifest_test)")
    # Option B: manifest-based
    p.add_argument("--manifest_test", type=str, default=None, help="Test manifest CSV/JSON (if not using --data_dir)")
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--use_multifeature", action="store_true")
    p.add_argument("--fusion_type", type=str, default="weighted")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./eval_outputs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.data_dir and not args.manifest_test:
        raise ValueError("Provide either --data_dir (e.g. data/fold1) or --manifest_test")

    os.makedirs(args.output_dir, exist_ok=True)
    processor = get_processor()
    transform = get_default_image_transform()

    model = load_model(
        args.checkpoint,
        use_multifeature=args.use_multifeature,
        device=args.device,
        fusion_type=args.fusion_type,
    )

    if args.data_dir:
        ds = BreastUSFolderDataset(
            data_dir=args.data_dir,
            split="test",
            image_transform=transform,
            use_multifeature=args.use_multifeature,
            require_all_modalities=False,
        )
    else:
        ds = BreastUSDataset(
            manifest_path=args.manifest_test,
            image_transform=transform,
            use_multifeature=args.use_multifeature,
            root_dir=args.root_dir,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor=processor),
    )

    print(f"Test set size: {len(ds)}")

    keyword_embeds = encode_keywords(model, processor, args.device)

    all_embeds = []
    all_labels = []
    all_distances = []

    with torch.no_grad():
        for batch in loader:
            if args.use_multifeature:
                img_embeds = model.encode_image_multifeature(
                    batch["pixel_values_bmode"].to(args.device),
                    batch["pixel_values_enhanced"].to(args.device),
                    batch["pixel_values_improved"].to(args.device),
                )
            else:
                img_embeds = model.encode_image_single(batch["pixel_values"].to(args.device))

            all_embeds.append(img_embeds.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

            # Distance from each image to each keyword (L2 or cosine distance = 1 - cos_sim for normalized)
            # Cosine similarity: (N, 512) @ (512, K) -> (N, K); distance = 1 - sim
            sim = img_embeds @ keyword_embeds.t()
            dist = 1.0 - sim.cpu().numpy()
            all_distances.append(dist)

    embeds = np.concatenate(all_embeds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    distances = np.concatenate(all_distances, axis=0)

    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeds)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "keyword_distances.npy"), distances)
    np.save(os.path.join(args.output_dir, "keyword_embeddings.npy"), keyword_embeds.cpu().numpy())

    # Summary: mean distance per class to each keyword
    summary = {}
    for kw_idx, kw in enumerate(KEYWORDS):
        summary[kw] = {}
        for cls_idx, cls_name in enumerate(BREAST_US_CLASSES):
            mask = labels == cls_idx
            if mask.sum() > 0:
                summary[kw][cls_name] = float(np.mean(distances[mask, kw_idx]))
            else:
                summary[kw][cls_name] = None
    with open(os.path.join(args.output_dir, "keyword_distance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved embeddings {embeds.shape}, labels {labels.shape}, distances {distances.shape} to {args.output_dir}")
    print("Keyword distance summary (mean distance per class):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
