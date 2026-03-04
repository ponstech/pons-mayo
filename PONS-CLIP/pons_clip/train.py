"""
Training script for PONS-CLIP: B-mode only or multi-feature (3 images per scan).

Usage:
  python -m pons_clip.train --config configs/bmode_only.yaml
  python -m pons_clip.train --config configs/multifeature.yaml
"""

import argparse
import os
from pathlib import Path

from .warnings_config import setup_warnings
setup_warnings()

import torch
from torch.utils.data import DataLoader

from .model import PONSCLIPModel
from .dataset import BreastUSDataset, BreastUSFolderDataset, collate_fn, get_default_image_transform
from .constants import BREAST_US_CLASSES


def get_processor():
    try:
        from medclip import MedCLIPProcessor
        return MedCLIPProcessor()
    except ImportError as e:
        raise ImportError(
            "PONS-CLIP requires MedCLIPProcessor from the medclip package for training. "
            "Install with: pip install medclip (or pip install git+https://github.com/RyanWangZf/MedCLIP.git). "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to load MedCLIPProcessor. Training cannot fall back to another tokenizer. "
            f"Original error: {e}"
        ) from e


def parse_args():
    p = argparse.ArgumentParser(description="Train PONS-CLIP")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    # Option A: folder-based (data/fold1 with train/val/test, each with bmode/enhanced/improved/benign|malignant)
    p.add_argument("--data_dir", type=str, default=None, help="e.g. data/fold1 — use train/ and val/ under it (overrides manifest)")
    # Option B: manifest-based
    p.add_argument("--manifest_train", type=str, default=None, help="Path to train manifest CSV/JSON (if not using --data_dir)")
    p.add_argument("--manifest_val", type=str, default=None, help="Path to val manifest")
    p.add_argument("--root_dir", type=str, default=None, help="Root for relative image paths (manifest mode)")
    p.add_argument("--use_multifeature", action="store_true", help="Use 3 images per scan (bmode, enhanced, improved)")
    p.add_argument("--fusion_type", type=str, default="weighted", choices=["weighted", "mlp", "attention"])
    p.add_argument("--output_dir", type=str, default="./outputs", help="Checkpoint and log dir")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    # Config values override defaults; CLI overrides config
    data_dir = args.data_dir or config.get("data_dir")
    manifest_train = args.manifest_train or config.get("manifest_train")
    manifest_val = args.manifest_val or config.get("manifest_val")
    if not data_dir and not manifest_train:
        raise ValueError("Provide either --data_dir (e.g. data/fold1) or --manifest_train")
    use_multifeature = args.use_multifeature or config.get("use_multifeature", False)
    fusion_type = args.fusion_type or config.get("fusion_type", "weighted")
    output_dir = args.output_dir or config.get("output_dir", "./outputs")
    epochs = args.epochs or config.get("epochs", 10)
    batch_size = args.batch_size or config.get("batch_size", 16)
    lr = args.lr if args.lr is not None else config.get("lr", 1e-5)
    device = args.device

    os.makedirs(output_dir, exist_ok=True)
    transform = get_default_image_transform()
    processor = get_processor()

    if data_dir:
        train_ds = BreastUSFolderDataset(
            data_dir=data_dir,
            split="train",
            image_transform=transform,
            use_multifeature=use_multifeature,
            require_all_modalities=False,
        )
        val_ds = BreastUSFolderDataset(
            data_dir=data_dir,
            split="val",
            image_transform=transform,
            use_multifeature=use_multifeature,
            require_all_modalities=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, processor=processor),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, processor=processor),
        )
        test_ds = BreastUSFolderDataset(
            data_dir=data_dir,
            split="test",
            image_transform=transform,
            use_multifeature=use_multifeature,
            require_all_modalities=False,
        )
        n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
        print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test} -> total={n_train + n_val + n_test}")
        print(f"Folder data: {data_dir}")
    else:
        train_ds = BreastUSDataset(
            manifest_path=manifest_train,
            image_transform=transform,
            use_multifeature=use_multifeature,
            root_dir=args.root_dir or config.get("root_dir"),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, processor=processor),
        )
        val_loader = None
        n_val = 0
        if manifest_val and os.path.isfile(manifest_val):
            val_ds = BreastUSDataset(
                manifest_path=manifest_val,
                image_transform=transform,
                use_multifeature=use_multifeature,
                root_dir=args.root_dir or config.get("root_dir"),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda b: collate_fn(b, processor=processor),
            )
            n_val = len(val_ds)
        print(f"Dataset sizes (manifest): train={len(train_ds)}, val={n_val} (test not loaded in train)")

    from medclip import MedCLIPVisionModelViT
    model = PONSCLIPModel(
        vision_cls=MedCLIPVisionModelViT,
        use_multifeature=use_multifeature,
        fusion_type=fusion_type,
    )
    model.from_pretrained()
    model = model.to(device)

    # Both modes finetune vision + text; multi-feature adds the fusion module
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if use_multifeature:
        print("Training mode: MULTI-FEATURE -> finetuning vision + text + fusion module")
    else:
        print("Training mode: B-MODE ONLY -> finetuning vision + text encoder")
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch.get("attention_mask"),
                "return_loss": True,
            }
            if use_multifeature:
                inputs["pixel_values_bmode"] = batch["pixel_values_bmode"].to(device)
                inputs["pixel_values_enhanced"] = batch["pixel_values_enhanced"].to(device)
                inputs["pixel_values_improved"] = batch["pixel_values_improved"].to(device)
            else:
                inputs["pixel_values"] = batch["pixel_values"].to(device)

            out = model(**inputs)
            loss = out["loss_value"]
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            nv = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch.get("attention_mask"),
                        "return_loss": True,
                    }
                    if use_multifeature:
                        inputs["pixel_values_bmode"] = batch["pixel_values_bmode"].to(device)
                        inputs["pixel_values_enhanced"] = batch["pixel_values_enhanced"].to(device)
                        inputs["pixel_values_improved"] = batch["pixel_values_improved"].to(device)
                    else:
                        inputs["pixel_values"] = batch["pixel_values"].to(device)
                    out = model(**inputs)
                    val_loss += out["loss_value"].item()
                    nv += 1
            print(f"  val_loss={val_loss/max(nv,1):.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(output_dir, f"pons_clip_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "use_multifeature": use_multifeature,
                "fusion_type": fusion_type,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    final_path = os.path.join(output_dir, "pons_clip_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "use_multifeature": use_multifeature,
        "fusion_type": fusion_type,
    }, final_path)
    print(f"Done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
