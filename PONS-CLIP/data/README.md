# Data directory

PONS-CLIP supports **folder-based** data (recommended for the standard fold layout) or **manifest-based** (CSV/JSON).

---

## Folder-based layout

One fold at a time, e.g. `data/fold1`. Expected structure:

```
data/fold1/
├── train/
│   ├── bmode/
│   │   ├── benign/    ← PNG files
│   │   └── malignant/
│   ├── enhanced/
│   │   ├── benign/
│   │   └── malignant/
│   └── improved/
│       ├── benign/
│       └── malignant/
├── val/
│   ├── bmode/benign/, bmode/malignant/
│   ├── enhanced/benign/, enhanced/malignant/
│   └── improved/benign/, improved/malignant/
└── test/
    ├── bmode/benign/, bmode/malignant/
    ├── enhanced/benign/, enhanced/malignant/
    └── improved/benign/, improved/malignant/
```

**Filename matching (per scan):**

- Scan list is defined by B-mode filenames under `bmode/benign/` and `bmode/malignant/`.
- For a B-mode file `bmode/benign/scan001.png`:
  - **Enhanced**: `enhanced/benign/scan001.png` or `enhanced/benign/scan001_enhanced.png`
  - **Improved**: `improved/benign/scan001.png` or `improved/benign/scan001_improved.png`
- If enhanced or improved is missing for any B-mode scan when using multi-feature, training exits with an error (no fallback).

**Example commands:**

```bash
# B-mode only
python -m pons_clip.train --data_dir data/fold1 --output_dir outputs_bmode

# Multi-feature
python -m pons_clip.train --data_dir data/fold1 --use_multifeature --output_dir outputs_multifeature

# Evaluate on fold1 test set
python -m pons_clip.evaluate --checkpoint outputs_bmode/pons_clip_final.pt --data_dir data/fold1 --output_dir eval_bmode
python -m pons_clip.evaluate --checkpoint outputs_multifeature/pons_clip_final.pt --data_dir data/fold1 --use_multifeature --output_dir eval_multifeature
```

---

## Manifest-based

CSV or JSON with columns:

| Column         | Required | Description |
|----------------|----------|-------------|
| `image_path` or `bmode_path` | Yes | Path to B-mode image |
| `enhanced_path` | For multi-feature | Path to enhanced image |
| `improved_path` | For multi-feature | Path to improved image |
| `label`        | Yes | `0` or `benign` / `1` or `malignant` |
| `text`         | No | Override description |

Use `--manifest_train`, `--manifest_val`, `--manifest_test` instead of `--data_dir`.
