"""
Prompt-based classification on the test set (no extra training).

Uses PONSCLIPPromptClassifier with breast US class prompts to get benign/malignant
logits per sample. Reports accuracy, AUC, sensitivity, specificity.
Run on existing checkpoints (B-mode or multi-feature) without retraining.
"""

import argparse
import json
import os

from .warnings_config import setup_warnings
setup_warnings()

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import PONSCLIPModel, PONSCLIPPromptClassifier
from .dataset import BreastUSFolderDataset, BreastUSDataset, collate_fn, get_default_image_transform
from .constants import BREAST_US_CLASSES
from .prompts import process_class_prompts, get_processor


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
    clf = PONSCLIPPromptClassifier(model, ensemble=True)
    return clf.to(device).eval()


def main():
    p = argparse.ArgumentParser(description="Prompt-based classification on test set (no training).")
    p.add_argument("--checkpoint", type=str, required=True, help="PONS-CLIP checkpoint")
    p.add_argument("--data_dir", type=str, default=None, help="e.g. data/fold1 (test/ under it)")
    p.add_argument("--manifest_test", type=str, default=None, help="Test manifest if not using data_dir")
    p.add_argument("--use_multifeature", action="store_true")
    p.add_argument("--fusion_type", type=str, default="weighted")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default=None, help="Write metrics JSON here (default: print only)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.data_dir and not args.manifest_test:
        raise ValueError("Provide --data_dir or --manifest_test")

    processor = get_processor()
    prompt_inputs = process_class_prompts(processor=processor)
    transform = get_default_image_transform()

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
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor=processor),
    )

    clf = load_model(args.checkpoint, args.use_multifeature, args.device, args.fusion_type)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if args.use_multifeature:
                out = clf(
                    pixel_values_bmode=batch["pixel_values_bmode"].to(args.device),
                    pixel_values_enhanced=batch["pixel_values_enhanced"].to(args.device),
                    pixel_values_improved=batch["pixel_values_improved"].to(args.device),
                    prompt_inputs=prompt_inputs,
                )
            else:
                out = clf(
                    pixel_values=batch["pixel_values"].to(args.device),
                    prompt_inputs=prompt_inputs,
                )
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = logits.argmax(axis=1)

    accuracy = (preds == labels).mean()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        prob_malignant = probs[:, 1]
        auc = roc_auc_score(labels, prob_malignant)
    except Exception:
        auc = None

    metrics = {
        "n_test": int(len(labels)),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc) if auc is not None else None,
        "class_names": BREAST_US_CLASSES,
    }

    print("Prompt-based classification (test set):")
    print(json.dumps(metrics, indent=2))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "prompt_classify_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
