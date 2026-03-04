"""PONS-CLIP: MedCLIP + multi-feature breast ultrasound fusion."""

from .constants import (
    BREAST_US_CLASSES,
    KEYWORDS,
    CLASS_DESCRIPTIONS,
    CLASS_PROMPTS,
    IMAGE_FEATURE_NAMES,
    NUM_IMAGE_FEATURES,
    PROJ_DIM,
)
from .dataset import BreastUSDataset, BreastUSFolderDataset, get_default_image_transform, collate_fn
from .fusion import FusionModule
from .model import PONSCLIPModel, PONSCLIPPromptClassifier
from .prompts import process_class_prompts

__all__ = [
    "PONSCLIPModel",
    "PONSCLIPPromptClassifier",
    "FusionModule",
    "BreastUSDataset",
    "BreastUSFolderDataset",
    "get_default_image_transform",
    "collate_fn",
    "process_class_prompts",
    "BREAST_US_CLASSES",
    "KEYWORDS",
    "CLASS_DESCRIPTIONS",
    "CLASS_PROMPTS",
    "IMAGE_FEATURE_NAMES",
    "NUM_IMAGE_FEATURES",
    "PROJ_DIM",
]
