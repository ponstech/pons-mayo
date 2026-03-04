"""Constants for PONS-CLIP: breast ultrasound and keyword setup."""

# Breast US classes (for labels and prompts)
BREAST_US_CLASSES = ["benign", "malignant"]

# Keywords for distance analysis (embedding space distance from sample embeddings)
KEYWORDS = [
    "benign",
    "malignant",
    "ultrasound image",
    "breast ultrasound",
    "normal",
    "abnormal",
    "mass",
    "lesion",
]

# Simple text descriptions per class (for contrastive training; replace with reports later)
CLASS_DESCRIPTIONS = {
    "benign": "benign breast ultrasound image",
    "malignant": "malignant breast ultrasound image",
}

# Default prompt templates for zero-shot style evaluation
CLASS_PROMPTS = {
    "benign": ["benign breast ultrasound image", "benign finding on ultrasound"],
    "malignant": ["malignant breast ultrasound image", "malignant finding on ultrasound"],
}

# Image feature types per scan
IMAGE_FEATURE_NAMES = ["bmode", "enhanced", "improved"]
NUM_IMAGE_FEATURES = 3

# Embedding dimension (MedCLIP projection dim)
PROJ_DIM = 512
