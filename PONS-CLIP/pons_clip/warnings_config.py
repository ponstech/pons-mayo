"""
Configure warnings for PONS-CLIP entry points.

These warnings come from dependencies (huggingface_hub, torch, transformers, umap),
not from PONS-CLIP code. Filtering reduces noise; causes are documented below.
"""

import warnings


def setup_warnings():
    """
    Suppress known third-party warnings that are expected or harmless.

    - huggingface_hub: FutureWarning about resume_download (deprecated API; HF will remove it).
    - torch: UserWarning about torch.meshgrid requiring 'indexing' (called inside Swin/vision code).
    - transformers: "Some weights ... were not used" when loading BERT/Swin for a different head (expected).
    - umap: "n_jobs value overridden by random_state" when using random_state for reproducibility.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*resume_download.*",
        category=FutureWarning,
        module="huggingface_hub",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*torch.meshgrid.*indexing.*",
        category=UserWarning,
        module="torch",
    )
    # Can be emitted from various transformers submodules
    warnings.filterwarnings(
        "ignore",
        message=".*Some weights of the model checkpoint.*were not used.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*n_jobs value.*overridden.*random_state.*",
        category=UserWarning,
        module="umap",
    )
