"""
Image–text similarity demo (CLIP-style): one image (or 3 for multi-feature) and a list of texts.

Usage:
  python -m pons_clip.similarity_demo --checkpoint ./outputs_bmode/pons_clip_final.pt --image path/to.png --texts "benign" "malignant" "ultrasound image"
  python -m pons_clip.similarity_demo --checkpoint ./outputs_multifeature/pons_clip_final.pt --use_multifeature --bmode path.png --enhanced path_enh.png --improved path_imp.png --texts "benign" "malignant"
"""

import argparse
import os

from .warnings_config import setup_warnings
setup_warnings()

import torch
from PIL import Image

from .model import PONSCLIPModel, PONSCLIPPromptClassifier
from .dataset import get_default_image_transform


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, default=None, help="Single image (B-mode only)")
    p.add_argument("--bmode", type=str, default=None, help="B-mode image (for multi-feature)")
    p.add_argument("--enhanced", type=str, default=None, help="Enhanced image (for multi-feature)")
    p.add_argument("--improved", type=str, default=None, help="Improved image (for multi-feature)")
    p.add_argument("--texts", type=str, nargs="+", default=["benign", "malignant", "ultrasound image"])
    p.add_argument("--use_multifeature", action="store_true")
    p.add_argument("--fusion_type", type=str, default="weighted")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    transform = get_default_image_transform()
    processor = get_processor()

    if args.use_multifeature:
        bmode_path = args.bmode or args.image
        if not bmode_path or not args.enhanced or not args.improved:
            raise ValueError("For multi-feature provide --bmode, --enhanced, --improved")
        img_b = transform(Image.open(bmode_path).convert("RGB")).unsqueeze(0)
        img_e = transform(Image.open(args.enhanced).convert("RGB")).unsqueeze(0)
        img_i = transform(Image.open(args.improved).convert("RGB")).unsqueeze(0)
        pixel_values_bmode = img_b.to(args.device)
        pixel_values_enhanced = img_e.to(args.device)
        pixel_values_improved = img_i.to(args.device)
    else:
        path = args.image or args.bmode
        if not path:
            raise ValueError("Provide --image (or --bmode)")
        img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        pixel_values = img.to(args.device)

    if hasattr(processor, "tokenizer"):
        enc = processor.tokenizer(args.texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    else:
        enc = processor(args.texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    model = load_model(args.checkpoint, args.use_multifeature, args.device, args.fusion_type)

    with torch.no_grad():
        if args.use_multifeature:
            img_embeds = model.encode_image_multifeature(pixel_values_bmode, pixel_values_enhanced, pixel_values_improved)
        else:
            img_embeds = model.encode_image_single(pixel_values)
        text_embeds = model.encode_text(input_ids, attention_mask)
        logits = model.medclip.compute_logits(img_embeds, text_embeds)
        sim = (img_embeds @ text_embeds.t()).cpu().numpy().flatten()

    print("Image–text similarity (cosine):")
    for t, s in zip(args.texts, sim):
        print(f"  {t!r}: {s:.4f}")
    print("(Higher = more similar)")


if __name__ == "__main__":
    main()
