"""
Retrieve the most similar scan(s) to a text prompt from the test set, or from
train + val + test. Copies retrieved images into an output folder labeled by
rank, ground-truth label, and modality (bmode, enhanced, improved).

Path resolution for enhanced/improved: the dataset tries exact filename first,
then stem_enhanced.png / stem_improved.png (see _find_enhanced_or_improved in dataset.py).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from .warnings_config import setup_warnings
setup_warnings()

import torch
import numpy as np
from torch.utils.data import DataLoader

from .model import PONSCLIPModel
from .dataset import BreastUSFolderDataset, get_default_image_transform, collate_fn
from .constants import BREAST_US_CLASSES


def get_processor():
    try:
        from medclip import MedCLIPProcessor
        return MedCLIPProcessor()
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


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


def _resolve_path_for_copy(path: Path, modality: str) -> Path:
    """Return path to copy from. Dataset resolves exact name vs _enhanced/_improved; retries resolution if the given path is missing."""
    if path.is_file():
        return path
    base_stem = path.stem.replace("_enhanced", "").replace("_improved", "")
    ext = path.suffix or ".png"
    if modality == "enhanced":
        for candidate in [path.parent / f"{base_stem}_enhanced{ext}", path.parent / f"{base_stem}{ext}"]:
            if candidate.is_file():
                return candidate
    elif modality == "improved":
        for candidate in [path.parent / f"{base_stem}_improved{ext}", path.parent / f"{base_stem}{ext}"]:
            if candidate.is_file():
                return candidate
    return path


def _copy_retrieved_to_output(match: dict, rank: int, output_dir: str) -> list:
    """Copy bmode, enhanced, improved to output_dir with names rank{N}_{label}_{modality}.ext. Returns list of copied paths."""
    label = match["ground_truth_label"]
    copied = []
    for modality, key in [("bmode", "bmode_path"), ("enhanced", "enhanced_path"), ("improved", "improved_path")]:
        src = Path(match[key])
        src = _resolve_path_for_copy(src, modality)
        if not src.is_file():
            continue
        ext = src.suffix or ".png"
        dest_name = "rank{}_{}_{}{}".format(rank, label, modality, ext)
        dest = Path(output_dir) / dest_name
        try:
            shutil.copy2(src, dest)
            copied.append(str(dest))
        except Exception as e:
            print("  Warning: could not copy {} -> {}: {}".format(src, dest, e))
    return copied


def main():
    p = argparse.ArgumentParser(description="Retrieve most similar scan(s) to a text prompt; copy to output folder.")
    p.add_argument("--prompt", type=str, required=True, help='e.g. "malignant ultrasound image with asymmetric lesion"')
    p.add_argument("--data_dir", type=str, required=True, help="e.g. data/fold1 (train/, val/, test/ under it)")
    p.add_argument("--checkpoint", type=str, required=True, help="PONS-CLIP checkpoint (B-mode or multi-feature)")
    p.add_argument("--use_multifeature", action="store_true", help="Use multi-feature model and 3 images per scan")
    p.add_argument("--fusion_type", type=str, default="weighted")
    p.add_argument("--splits", type=str, default="all", choices=["test", "all"],
                   help="Search in 'test' only or 'all' (train + val + test)")
    p.add_argument("--top_k", type=int, default=1, help="Number of top matches to return")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="./retrieval_output",
                   help="Folder where retrieved images are copied (rank{N}_{label}_{modality}.png) and JSON is saved")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    processor = get_processor()
    transform = get_default_image_transform()

    if args.splits == "test":
        split_names = ["test"]
    else:
        split_names = ["train", "val", "test"]

    datasets = {}
    all_embeds = []
    index_to_split_and_idx = []

    model = load_model(args.checkpoint, args.use_multifeature, args.device, args.fusion_type)

    for split in split_names:
        ds = BreastUSFolderDataset(
            data_dir=args.data_dir,
            split=split,
            image_transform=transform,
            use_multifeature=args.use_multifeature,
            require_all_modalities=False,
        )
        datasets[split] = ds
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, processor=processor),
        )
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

    img_embeds = np.concatenate(all_embeds, axis=0)
    index_to_split_and_idx = []
    for split in split_names:
        for local_idx in range(len(datasets[split])):
            index_to_split_and_idx.append((split, local_idx))

    # Encode prompt
    if hasattr(processor, "tokenizer"):
        enc = processor.tokenizer(
            [args.prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
    else:
        enc = processor(
            [args.prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    with torch.no_grad():
        prompt_embed = model.encode_text(input_ids, attention_mask)
    prompt_vec = prompt_embed.cpu().numpy()
    similarities = (img_embeds @ prompt_vec.T).flatten()

    top_global_indices = np.argsort(similarities)[::-1][: args.top_k]

    results = {
        "prompt": args.prompt,
        "checkpoint": args.checkpoint,
        "use_multifeature": args.use_multifeature,
        "data_dir": args.data_dir,
        "splits_searched": split_names,
        "total_scans": int(len(img_embeds)),
        "top_k": args.top_k,
        "matches": [],
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for rank, global_idx in enumerate(top_global_indices, start=1):
        split, local_idx = index_to_split_and_idx[global_idx]
        ds = datasets[split]
        s = ds.samples[local_idx]
        sim = float(similarities[global_idx])
        match = {
            "rank": rank,
            "similarity": round(sim, 6),
            "split": split,
            "ground_truth_label": s["label_name"],
            "bmode_path": str(s["bmode_path"]),
            "enhanced_path": str(s["enhanced_path"]),
            "improved_path": str(s["improved_path"]),
        }
        results["matches"].append(match)

        copied = _copy_retrieved_to_output(match, rank, args.output_dir)
        match["copied_to"] = copied

    print("Prompt:", args.prompt)
    print("Model:", "multi-feature" if args.use_multifeature else "B-mode only")
    print("Searched splits:", split_names, "| total scans:", len(img_embeds))
    print("Top {} match(es) (copied to {}):".format(args.top_k, args.output_dir))
    for m in results["matches"]:
        print("  rank {} | similarity {:.4f} | split {} | label {} | bmode {} | enhanced {} | improved {}".format(
            m["rank"], m["similarity"], m["split"], m["ground_truth_label"],
            m["bmode_path"], m["enhanced_path"], m["improved_path"],
        ))

    out_path = os.path.join(args.output_dir, "retrieve_by_prompt.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out_path)

    return results


if __name__ == "__main__":
    main()
