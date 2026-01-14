# NoisyNN Multi-Feature Fusion for Breast Cancer Classification

This repository provides an implementation of the **Noisy Neural Network (NoisyNN)** framework combined with **multi-feature fusion strategies (Early, Mid, and Late Fusion)** for **breast cancer classification**. The model leverages multi-scale convolutional neural networks and noise injection to improve robustness and generalization.
---
## Overview
The proposed approach performs breast cancer classification using multiple representations of the same image. Each representation is processed using multi-scale CNN branches, and different fusion strategies are explored to analyze their effect on classification performance.

Noise injection based on the NoisyNN framework can be applied at configurable layers to improve model robustness.
---

## Features

- **Early Fusion**: Fusion of multiple image representations at the input level
- **Mid Fusion**: Fusion of features at an intermediate network layer
- **Late Fusion**: Fusion of features at the final classification layer
- **NoisyNN Support**:
  - Linear noise
  - Gaussian noise
  - Impulse noise
- **Multi-Scale ResNet Backbone**:
  - 3×3 convolution kernels
  - 5×5 convolution kernels
  - 7×7 convolution kernels
- **3-Fold Cross-Validation Support**

---

## Dataset Structure

The dataset must be organized as follows:

```
combined-breast-3fold/
├── fold1/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── fold2/
└── fold3/
```

## Usage

### Parameter Settings

You can configure the following parameters in the `parameters.py` file:

- `--breast_dataset_path`: Path to the dataset directory (default: `../combined-breast-3fold`)
- `--fold`: Fold number to be used (1, 2, or 3) (default: 1)
- `--fusion_type`: Fusion strategy (`early`, `mid`, or `late`) (default: `mid`)
- `--noise_type`: Noise type (`linear`, `gaussian`, `impulse`) (default: `impulse`)
- `--noise_str`: Noise strength (default: 0.1)
- `--noise_layer`: Layer where noise is injected (1–4) (default: 4)
- `--batch_size`: Batch size (default: 128)
- `--epoch`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)

---

### Training

Training with **early fusion**:

```bash
cd NoisyNN-main/NoisyCNN_CircularShiftQ
python Main_Fusion.py --fusion_type early --fold 1 --noise_type linear --noise_str 0.1 --noise_layer 2
``` 

Training with **mid fusion**:

```bash
cd NoisyNN-main/NoisyCNN_CircularShiftQ
python Main_Fusion.py --fusion_type mid --fold 1 --noise_type linear --noise_str 0.1 --noise_layer 2
```

Training with **late fusion**:

```bash
python Main_Fusion.py --fusion_type late --fold 1 --noise_type impulse --noise_str 0.1 --noise_layer 4
```

### Example Commands

**Mid Fusion, Fold 1, Linear Noise:**
```bash
python Main_Fusion.py \
    --fusion_type mid \
    --fold 1 \
    --noise_type linear \
    --noise_str 0.1 \
    --noise_layer 2 \
    --batch_size 32 \
    --epoch 100 \
    --lr 0.001
```

## Model Architecture

### Early Fusion
- Five different image representations are fused at the input level
- The fused input is processed by a shared convolutional backbone
- Multi-scale feature extraction using -->  3×3 kernels,  5×5 kernels and 5x5 kernels

### Mid Fusion
- Each of the five images is processed independently through the initial convolution layers
- Feature maps are fused at an intermediate layer (after max pooling)
- The fused features are passed through multi-scale branches
- Feature extraction is performed using 3×3, 5×5, and 7×7 kernel sizes

### Late Fusion
- Each of the five images is processed completely independently
- Multi-scale feature extraction is applied separately for each image
- All extracted features are fused at the final classification layer
- Uses a larger number of parameters (approximately 5× more features)

## Outputs

- Model checkpoints are saved in the saved_models/ directory
- Model naming format: `{accuracy}_breast_{fusion_type}_fold{fold}_noise_{noise_type}_str_{strength}_layer_{layer}.pth`

## Notes

- The dataset path must be set via `breast_dataset_path` in `parameters.py`
- Multi-feature için her görüntü 5 farklı augmentasyon ile işlenir
- NoisyNN noise injection is applied at the specified layer
- The model is automatically saved based on the best validation accuracy

