"""
PONS-CLIP model: MedCLIP with optional multi-feature fusion.

- B-mode only: one image per sample; vision + text encoders finetuned.
- Multi-feature: three images per sample (bmode, enhanced, improved); same
  vision encoder (finetuned) and text encoder (finetuned) plus a FusionModule
  that combines the three image embeddings. Vision and text are finetuned
  in both modes; the only difference is the fusion module in multi-feature.
"""

import torch
import torch.nn as nn
from medclip import MedCLIPModel as _MedCLIPModel
from medclip import MedCLIPVisionModelViT, MedCLIPVisionModel

from .fusion import FusionModule
from .constants import PROJ_DIM, NUM_IMAGE_FEATURES, BREAST_US_CLASSES


class PONSCLIPModel(nn.Module):
    """
    PONS-CLIP: MedCLIP with optional 3-image fusion.

    - use_multifeature=False (B-mode only): vision + text encoders finetuned.
    - use_multifeature=True: vision + text encoders finetuned, plus FusionModule
      to fuse bmode/enhanced/improved embeddings. No freezing; the only
      difference from B-mode is the fusion module.
    """

    def __init__(
        self,
        vision_cls=MedCLIPVisionModelViT,
        use_multifeature: bool = False,
        fusion_type: str = "weighted",
        checkpoint=None,
        vision_checkpoint=None,
        **medclip_kwargs,
    ):
        super().__init__()
        self.use_multifeature = use_multifeature
        self.medclip = _MedCLIPModel(
            vision_cls=vision_cls,
            checkpoint=checkpoint,
            vision_checkpoint=vision_checkpoint,
            **medclip_kwargs,
        )

        if use_multifeature:
            self.fusion = FusionModule(dim=PROJ_DIM, fusion_type=fusion_type)
        else:
            self.fusion = None

    def from_pretrained(self, input_dir=None):
        self.medclip.from_pretrained(input_dir=input_dir)

    def encode_image_single(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Single image -> L2-normalized 512-d (for b-mode only or per-view)."""
        img_embeds = self.medclip.vision_model(pixel_values=pixel_values)
        return img_embeds / (img_embeds.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_image_multifeature(
        self,
        pixel_values_bmode: torch.Tensor,
        pixel_values_enhanced: torch.Tensor,
        pixel_values_improved: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 3 views with vision encoder (finetuned), then fuse."""
        e_b = self.encode_image_single(pixel_values_bmode)
        e_e = self.encode_image_single(pixel_values_enhanced)
        e_i = self.encode_image_single(pixel_values_improved)
        stack = torch.stack([e_b, e_e, e_i], dim=1)
        return self.fusion(stack)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.medclip.encode_text(input_ids=input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_bmode=None,
        pixel_values_enhanced=None,
        pixel_values_improved=None,
        return_loss=True,
        **kwargs,
    ):
        """
        Args:
            For B-mode only: pixel_values (B, 3, H, W).
            For multi-feature: pixel_values_bmode, pixel_values_enhanced, pixel_values_improved
                each (B, 3, H, W).
            input_ids, attention_mask: text inputs.
        """
        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if self.use_multifeature and pixel_values_bmode is not None:
            pixel_values_bmode = pixel_values_bmode.to(device)
            pixel_values_enhanced = pixel_values_enhanced.to(device)
            pixel_values_improved = pixel_values_improved.to(device)
            img_embeds = self.encode_image_multifeature(
                pixel_values_bmode,
                pixel_values_enhanced,
                pixel_values_improved,
            )
        else:
            pv = pixel_values if pixel_values is not None else pixel_values_bmode
            img_embeds = self.encode_image_single(pv.to(device))

        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.medclip.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.medclip.clip_loss(logits_per_text)
        else:
            loss = None

        return {
            "img_embeds": img_embeds,
            "text_embeds": text_embeds,
            "logits": logits_per_image,
            "loss_value": loss,
            "logits_per_text": logits_per_text,
        }


class PONSCLIPPromptClassifier(nn.Module):
    """
    Prompt-based classifier for PONS-CLIP (like MedCLIP's PromptClassifier).
    Uses class prompts (e.g. benign / malignant) to get logits per class without training a head.
    Supports both B-mode only and multi-feature (3 images).
    """

    def __init__(self, model: PONSCLIPModel, ensemble: bool = True):
        super().__init__()
        self.model = model
        self.ensemble = ensemble
        self.class_names = BREAST_US_CLASSES

    def forward(
        self,
        pixel_values=None,
        pixel_values_bmode=None,
        pixel_values_enhanced=None,
        pixel_values_improved=None,
        prompt_inputs=None,
        **kwargs,
    ):
        """
        prompt_inputs: dict { class_name: {"input_ids": (n_prompts, L), "attention_mask": (n_prompts, L) } }
        """
        device = next(self.model.parameters()).device
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if pixel_values_bmode is not None:
            pixel_values_bmode = pixel_values_bmode.to(device)
            pixel_values_enhanced = pixel_values_enhanced.to(device)
            pixel_values_improved = pixel_values_improved.to(device)

        class_logits = []
        for cls_name in self.class_names:
            if cls_name not in prompt_inputs:
                continue
            inp = prompt_inputs[cls_name]
            input_ids = inp["input_ids"].to(device)
            attention_mask = inp.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_loss": False,
            }
            if self.model.use_multifeature and pixel_values_bmode is not None:
                inputs["pixel_values_bmode"] = pixel_values_bmode
                inputs["pixel_values_enhanced"] = pixel_values_enhanced
                inputs["pixel_values_improved"] = pixel_values_improved
            else:
                inputs["pixel_values"] = pixel_values
            out = self.model(**inputs)
            logits = out["logits"]
            if self.ensemble:
                cls_sim = logits.mean(dim=1)
            else:
                cls_sim = logits.max(dim=1)[0]
            class_logits.append(cls_sim)
        logits = torch.stack(class_logits, dim=1)
        return {"logits": logits, "class_names": self.class_names}
