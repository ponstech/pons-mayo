"""
Dataset for PONS-CLIP: breast US with optional enhanced/improved images.

Two ways to specify data:

1. **Folder-based** (recommended for the standard fold layout):
   - Base path: e.g. data/fold1
   - Splits: train, val, test under base
   - Under each split: bmode/benign, bmode/malignant, enhanced/benign, enhanced/malignant,
     improved/benign, improved/malignant, each containing PNGs.
   - Use BreastUSFolderDataset with data_dir="data/fold1", split="train" (or "val"/"test").
   - Enhanced/improved filenames can be same as bmode, or have "_enhanced" / "_improved" suffix.

2. **Manifest-based** (single CSV/JSON listing every sample):
   - Columns: image_path or bmode_path, optional enhanced_path, improved_path, label, optional text.
   - Use BreastUSDataset with manifest_path="path/to/manifest.csv".
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T

from .constants import CLASS_DESCRIPTIONS, BREAST_US_CLASSES, IMAGE_FEATURE_NAMES

# MedCLIP-style image normalization (match medclip.constants if available)
try:
    from medclip import constants as medclip_constants
    IMG_SIZE = getattr(medclip_constants, "IMG_SIZE", 224)
    IMG_MEAN = getattr(medclip_constants, "IMG_MEAN", 0.5862785803043838)
    IMG_STD = getattr(medclip_constants, "IMG_STD", 0.27950088968644304)
except Exception:
    IMG_SIZE, IMG_MEAN, IMG_STD = 224, 0.5862785803043838, 0.27950088968644304


def get_default_image_transform():
    """Transform PIL Image -> tensor for MedCLIP (resize, normalize)."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[IMG_MEAN] * 3, std=[IMG_STD] * 3),
    ])


def _resolve_path(root: str, path: str) -> str:
    if path is None or (isinstance(path, float) and pd.isna(path)):
        return None
    path = str(path).strip()
    if not os.path.isabs(path) and root:
        path = os.path.join(root, path)
    return path if os.path.isfile(path) else None


def _label_to_int(label) -> int:
    if isinstance(label, int):
        return int(label)
    s = str(label).strip().lower()
    if s in ("benign", "0"):
        return 0
    if s in ("malignant", "1"):
        return 1
    try:
        return int(label)
    except (ValueError, TypeError):
        return 0


class BreastUSDataset(Dataset):
    """
    Breast US dataset: one row per scan.
    - If enhanced_path and improved_path are present and use_multifeature=True,
      load 3 images; otherwise only B-mode.
    """

    def __init__(
        self,
        manifest_path: str,
        image_transform=None,
        use_multifeature: bool = False,
        root_dir: str = None,
        text_column: str = "text",
        label_column: str = "label",
    ):
        """
        manifest_path: path to CSV/JSON with columns image_path (or bmode_path),
            optional enhanced_path, improved_path, label, optional text.
        image_transform: callable PIL Image -> tensor (e.g. from MedCLIPProcessor).
        use_multifeature: if True, require enhanced_path and improved_path.
        root_dir: prefix for relative paths in manifest.
        """
        self.root_dir = root_dir or os.path.dirname(manifest_path)
        self.transform = image_transform
        self.use_multifeature = use_multifeature
        self.text_column = text_column
        self.label_column = label_column

        path = Path(manifest_path)
        if path.suffix.lower() == ".json":
            self.df = pd.read_json(manifest_path)
        else:
            self.df = pd.read_csv(manifest_path)

        # Normalize column names
        if "bmode_path" not in self.df.columns and "image_path" in self.df.columns:
            self.df["bmode_path"] = self.df["image_path"]
        if "image_path" not in self.df.columns and "bmode_path" in self.df.columns:
            self.df["image_path"] = self.df["bmode_path"]

        self.df["label_int"] = self.df[self.label_column].map(_label_to_int)
        self.class_names = BREAST_US_CLASSES

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        p = _resolve_path(self.root_dir, path)
        if p is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(p).convert("RGB")

    def _get_text(self, row) -> str:
        if self.text_column in row and pd.notna(row.get(self.text_column)) and str(row[self.text_column]).strip():
            return str(row[self.text_column]).strip()
        idx = row["label_int"]
        return CLASS_DESCRIPTIONS[self.class_names[idx]]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        bmode_path = row.get("bmode_path") or row.get("image_path")
        img_b = self._load_image(bmode_path)
        if self.transform:
            img_b = self.transform(img_b)

        out = {
            "pixel_values": img_b,
            "label": row["label_int"],
            "label_name": self.class_names[row["label_int"]],
        }

        if self.use_multifeature:
            enh_path = row.get("enhanced_path")
            imp_path = row.get("improved_path")
            if enh_path is None:
                raise FileNotFoundError(
                    f"Multi-feature training requires enhanced_path for every sample. "
                    f"Missing for B-mode: {bmode_path}"
                )
            if imp_path is None:
                raise FileNotFoundError(
                    f"Multi-feature training requires improved_path for every sample. "
                    f"Missing for B-mode: {bmode_path}"
                )
            img_e = self._load_image(enh_path)
            img_i = self._load_image(imp_path)
            if self.transform:
                img_e = self.transform(img_e)
                img_i = self.transform(img_i)
            out["pixel_values_bmode"] = out["pixel_values"]
            out["pixel_values_enhanced"] = img_e
            out["pixel_values_improved"] = img_i

        out["text"] = self._get_text(row)
        return out


def _find_enhanced_or_improved(base_dir: Path, class_name: str, bmode_stem: str, suffix: str) -> Optional[Path]:
    """Resolve path for enhanced or improved image. suffix is '_enhanced' or '_improved'."""
    # Same name as bmode
    p_same = base_dir / class_name / f"{bmode_stem}.png"
    if p_same.is_file():
        return p_same
    # With suffix: stem_enhanced.png or stem_improved.png
    p_suff = base_dir / class_name / f"{bmode_stem}{suffix}.png"
    if p_suff.is_file():
        return p_suff
    return None


class BreastUSFolderDataset(Dataset):
    """
    Breast US dataset from folder layout:

        data/fold1/{split}/   where split in train, val, test
            bmode/benign/*.png, bmode/malignant/*.png
            enhanced/benign/*.png, enhanced/malignant/*.png
            improved/benign/*.png, improved/malignant/*.png

    B-mode filenames define the set of scans. For each bmode file, enhanced and
    improved are looked up by:
      - same filename (e.g. scan001.png), or
      - stem + suffix (e.g. scan001_enhanced.png, scan001_improved.png).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        image_transform=None,
        use_multifeature: bool = False,
        require_all_modalities: bool = False,
    ):
        """
        data_dir: path to the fold root, e.g. data/fold1 (must contain train/, val/, test/).
        split: "train", "val", or "test".
        image_transform: callable PIL Image -> tensor.
        use_multifeature: if True, load bmode + enhanced + improved; training fails if any are missing.
        require_all_modalities: unused (all modalities required when use_multifeature=True).
        """
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.transform = image_transform
        self.use_multifeature = use_multifeature
        self.require_all_modalities = require_all_modalities
        self.class_names = BREAST_US_CLASSES

        split_dir = self.data_dir / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        bmode_dir = split_dir / "bmode"
        enhanced_dir = split_dir / "enhanced"
        improved_dir = split_dir / "improved"

        self.samples = []
        missing_enhanced = []
        missing_improved = []
        for class_name in self.class_names:
            label_int = self.class_names.index(class_name)
            class_bmode = bmode_dir / class_name
            if not class_bmode.is_dir():
                continue
            for png in sorted(class_bmode.glob("*.png")):
                stem = png.stem
                base_stem = stem.replace("_enhanced", "").replace("_improved", "")

                enh_path = _find_enhanced_or_improved(enhanced_dir, class_name, base_stem, "_enhanced")
                imp_path = _find_enhanced_or_improved(improved_dir, class_name, base_stem, "_improved")

                if self.use_multifeature:
                    if enh_path is None:
                        missing_enhanced.append(str(png.relative_to(split_dir)))
                    if imp_path is None:
                        missing_improved.append(str(png.relative_to(split_dir)))
                    if enh_path is None or imp_path is None:
                        continue
                else:
                    if enh_path is None:
                        enh_path = png
                    if imp_path is None:
                        imp_path = png

                self.samples.append({
                    "bmode_path": png,
                    "enhanced_path": enh_path,
                    "improved_path": imp_path,
                    "label": label_int,
                    "label_name": class_name,
                })

        if self.use_multifeature and (missing_enhanced or missing_improved):
            msg = (
                "Multi-feature training requires enhanced and improved images for every B-mode scan. "
                "No fallback is used; training is stopped.\n"
                "Expected: for each bmode/.../stem.png, have enhanced/.../stem.png or stem_enhanced.png, "
                "and improved/.../stem.png or stem_improved.png.\n"
            )
            if missing_enhanced:
                msg += f"Missing enhanced for {len(missing_enhanced)} scan(s), e.g. {missing_enhanced[:3]}"
                if len(missing_enhanced) > 3:
                    msg += f" ... and {len(missing_enhanced) - 3} more."
                msg += "\n"
            if missing_improved:
                msg += f"Missing improved for {len(missing_improved)} scan(s), e.g. {missing_improved[:3]}"
                if len(missing_improved) > 3:
                    msg += f" ... and {len(missing_improved) - 3} more."
            raise FileNotFoundError(msg)

        if len(self.samples) == 0:
            raise ValueError(f"No PNGs found under {split_dir}/bmode/benign and bmode/malignant")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        p = path if path.is_file() else None
        if p is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img_b = self._load_image(s["bmode_path"])
        if self.transform:
            img_b = self.transform(img_b)

        out = {
            "pixel_values": img_b,
            "label": s["label"],
            "label_name": s["label_name"],
            "text": CLASS_DESCRIPTIONS[s["label_name"]],
        }

        if self.use_multifeature:
            img_e = self._load_image(s["enhanced_path"])
            img_i = self._load_image(s["improved_path"])
            if self.transform:
                img_e = self.transform(img_e)
                img_i = self.transform(img_i)
            out["pixel_values_bmode"] = out["pixel_values"]
            out["pixel_values_enhanced"] = img_e
            out["pixel_values_improved"] = img_i

        return out


def collate_fn(batch: list, processor=None):
    """
    Collate batch and optionally tokenize text with processor.
    processor: MedCLIPProcessor; if provided, tokenizes batch text.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts = [b["text"] for b in batch]

    out = {
        "pixel_values": pixel_values,
        "labels": labels,
        "texts": texts,
    }

    if "pixel_values_bmode" in batch[0]:
        out["pixel_values_bmode"] = torch.stack([b["pixel_values_bmode"] for b in batch])
        out["pixel_values_enhanced"] = torch.stack([b["pixel_values_enhanced"] for b in batch])
        out["pixel_values_improved"] = torch.stack([b["pixel_values_improved"] for b in batch])

    if processor is not None:
        enc = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        out["input_ids"] = enc["input_ids"]
        out["attention_mask"] = enc.get("attention_mask")

    return out
