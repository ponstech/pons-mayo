"""
Build prompt inputs for PONS-CLIP prompt-based classification (breast US: benign / malignant).

Similar to MedCLIP's generate_chexpert_class_prompts / process_class_prompts,
but for our class set and compatible with any tokenizer that has a __call__(text=..., return_tensors="pt").
"""

from .constants import CLASS_PROMPTS, BREAST_US_CLASSES


def get_processor():
    try:
        from medclip import MedCLIPProcessor
        return MedCLIPProcessor()
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def process_class_prompts(processor=None, max_length: int = 77):
    """
    Turn CLASS_PROMPTS into tokenized inputs per class for PromptClassifier.

    Returns:
        dict: { "benign": {"input_ids": (n_prompts, L), "attention_mask": (n_prompts, L) }, "malignant": {...} }
    """
    if processor is None:
        processor = get_processor()
    prompt_inputs = {}
    for cls_name in BREAST_US_CLASSES:
        texts = CLASS_PROMPTS.get(cls_name, [f"{cls_name} breast ultrasound image"])
        if hasattr(processor, "tokenizer"):
            enc = processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        else:
            enc = processor(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        prompt_inputs[cls_name] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask"),
        }
    return prompt_inputs
