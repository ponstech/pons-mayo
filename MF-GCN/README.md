# Multi-Feature Graph Convolutional Network (MF-GCN)

A deep learning framework for ultrasound image classification using multi-feature fusion and graph convolutional networks. This implementation supports multiple backbone architectures (CustomCNN, ResNet18, DINO, MAE ViT, i-JEPA) and combines features from three different image representations (B-mode, Enhanced, Improved) using concatenation or MLP fusion, followed by graph-based classification.

## Overview

This project implements a Multi-Feature Graph Convolutional Network for binary classification of ultrasound images (benign vs. malignant). The architecture consists of four main stages:

1. **Feature Extraction**: Independent CNN/ViT backbones extract 512-dimensional features from each of three image types
2. **Feature Fusion**: Features are combined via concatenation (1536-D) or MLP fusion (256-D)
3. **Graph Construction**: Cosine similarity-based graph construction with top-K neighbor selection
4. **Graph Convolutional Network**: GCN-based node classification for final predictions

## Architecture Details

### Supported Backbone Architectures

- **CustomCNN**: Lightweight CNN with 3 convolutional layers (32, 64, 128 channels) + 2 fully connected layers
- **ResNet18**: Pre-trained ResNet18 from ImageNet with multi-scale feature extraction
- **DINO ViT-Small**: Vision Transformer pre-trained with DINO self-supervised learning (12 layers, 384 embed_dim)
- **MAE ViT-Base**: Masked Autoencoder Vision Transformer (12 layers, 768 embed_dim) - requires checkpoint file
- **i-JEPA ViT**: Joint Embedding Predictive Architecture from HuggingFace (configurable layers)

### Feature Fusion Methods

- **Concatenation**: Simple concatenation of three 512-D feature vectors → 1536-D output
- **MLP Fusion**: Two-layer MLP (1536 → 512 → 256) with batch normalization and dropout

### Graph Construction

- **Method**: Cosine similarity top-K
- **Normalization**: Z-score normalization
- **Top-K**: Default K=7 neighbors per node
- **Graph Type**: Undirected, sparse connectivity

### GCN Architecture

- **Model**: GCNNClassifier (default)
- **Layers**: 1 graph convolutional layer (256 hidden channels) + 1 classification layer
- **Output**: Class logits (2 classes: benign=0, malignant=1)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch 2.2 or higher

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Note**: For PyTorch Geometric with CUDA support, you may need to install additional dependencies. See [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

### Key Dependencies

- `torch>=2.2` and `torchvision>=0.17`
- `torch-geometric>=2.4`
- `timm>=1.0.3` (for DINO and MAE backbones)
- `transformers>=4.30.0` (for i-JEPA backbone)
- `scikit-learn>=1.3`, `pandas>=2.0`, `numpy>=1.24`
- `PyYAML>=6.0`, `matplotlib>=3.7`, `seaborn>=0.12`

## Data Structure

The code expects data organized in the following structure:

```
data/
└── combined_3folds_balanced/  # or your data directory
    ├── fold1/  # (optional, if using folds)
    │   ├── train/
    │   │   ├── bmode/
    │   │   │   ├── benign/
    │   │   │   │   ├── image1.png
    │   │   │   │   └── image2.png
    │   │   │   └── malignant/
    │   │   │       └── ...
    │   │   ├── enhanced/
    │   │   │   ├── benign/
    │   │   │   └── malignant/
    │   │   └── improved/
    │   │       ├── benign/
    │   │       └── malignant/
    │   ├── val/
    │   │   └── ... (same structure as train)
    │   └── test/
    │       └── ... (same structure as train)
    ├── fold2/
    └── fold3/
```

**Important**: 
- Images should be PNG or JPG format
- Each sample should have aligned images across all three feature types (bmode, enhanced, improved)
- If a feature type directory doesn't exist, the code will use bmode images as fallback (with a warning)

## Configuration

Edit `config.yaml` to configure the pipeline:

### Key Configuration Options

```yaml
# Data paths
base_data_dir: data/combined_3folds_balanced
modalities: [bmode, enhanced, improved]
folds: [fold1, fold2, fold3]

# Feature extraction
cnn:
  feature_dim: 512
  freeze_backbone: True
  train_last_n_layers: 1
  epochs: 50
  lr: 0.0001
  mae_checkpoint: /path/to/checkpoint-29.pth  # Required for MAE backbone

# Fusion method
fusion:
  method: concat  # or "mlp"

# Graph construction
graph:
  type: cosine_topk
  top_k: 7

# GCN training
gcn:
  model: GCNNClassifier
  hidden_channels: 256
  epochs: 125
  lr: 0.001
```

### MAE Checkpoint Setup

For MAE ViT backbone, you need to provide a fine-tuned checkpoint file:

1. Set the path in `config.yaml`:
   ```yaml
   cnn:
     mae_checkpoint: /path/to/your/checkpoint-29.pth
   ```

2. Or set environment variable:
   ```bash
   export MAE_CKPT=/path/to/your/checkpoint-29.pth
   ```

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python run_pipeline.py --config config.yaml
```

This will:
1. Train feature extractors for all backbones (custom, resnet18, dino_vits8, mae_vit, ijepa_vit)
2. Extract features for each feature type (bmode, enhanced, improved)
3. Fuse features (concatenation or MLP)
4. Build graphs
5. Train GCN models
6. Evaluate on test sets

### Command-Line Options

```bash
python run_pipeline.py \
    --config config.yaml \
    --backbones resnet18 mae_vit \  # Train only specific backbones
    --folds fold1 fold2 \            # Process only specific folds
    --device cuda \                  # Specify device (cuda/cpu)
    --output_dir artifacts           # Override output directory
```

### Training Specific Backbones

Train only selected backbones:

```bash
# Train only ResNet18 and MAE
python run_pipeline.py --backbones resnet18 mae_vit

# Train only CustomCNN
python run_pipeline.py --backbones custom
```

### Processing Specific Folds

Process only certain folds:

```bash
python run_pipeline.py --folds fold1 fold2
```

## Output Structure

Results are saved in the `artifacts/` directory (or as specified in config):

```
artifacts/
├── {backbone_name}/
│   ├── config_{run_id}.yaml          # Configuration snapshot
│   ├── cnn/                           # Feature extractor results
│   │   ├── bmode/
│   │   │   └── best_{backbone}_in{channels}.pth
│   │   ├── enhanced/
│   │   └── improved/
│   ├── features/                      # Extracted features (CSV)
│   │   ├── bmode_train.csv
│   │   ├── bmode_val.csv
│   │   ├── bmode_test.csv
│   │   └── ... (same for enhanced, improved)
│   ├── fusion/                        # Fused features
│   │   ├── train_features_concat.csv
│   │   ├── val_features_concat.csv
│   │   └── test_features_concat.csv
│   ├── gcn/                           # GCN model and results
│   │   ├── best_gcn_{model_name}.pth
│   │   ├── gcn_loss.png
│   │   └── gcn_metrics_concat.csv
│   └── plots/                         # Visualizations
│       ├── cnn_cm_{modality}.png
│       └── gcn_cm_concat.png
```

## Pipeline Stages

The pipeline can be run in stages by modifying `config.yaml`:

```yaml
run:
  stages: [cnn, gcn]  # Run both stages
  # stages: [cnn]     # Only train feature extractors
  # stages: [gcn]     # Only train GCN (requires existing features)
```

### Stage 1: Feature Extraction (CNN)

- Trains separate feature extractors for each feature type (bmode, enhanced, improved)
- Each extractor is trained independently
- Saves features to CSV files

### Stage 2: Feature Fusion

- Loads features from all three feature types
- Applies fusion method (concatenation or MLP)
- Saves fused features

### Stage 3: Graph Construction

- Builds graphs using cosine similarity top-K method
- Creates separate graphs for train/val/test splits

### Stage 4: GCN Training

- Trains GCN on training graph
- Validates on validation graph
- Evaluates on test graph
- Generates predictions and metrics

## Model Architecture Details

### Feature Extractors

Each feature type (B-mode, Enhanced, Improved) is processed through a separate instance of the selected backbone architecture. The three extractors are trained independently (no weight sharing).

**Output**: 512-dimensional feature vectors per feature type

### Feature Fusion

**Concatenation Fusion** (Default):
- Input: 3 × 512-D vectors
- Output: 1536-D concatenated vector
- No learnable parameters

**MLP Fusion**:
- Input: 1536-D (concatenated)
- Hidden Layer 1: 1536 → 512 (with batch norm, ReLU, dropout 0.5)
- Hidden Layer 2: 512 → 256 (with batch norm, ReLU, dropout 0.5)
- Output: 256-D fused vector

### Graph Construction

1. **Normalization**: Z-score normalization of all features
2. **Similarity Computation**: Cosine similarity between all pairs of feature vectors
3. **Edge Creation**: For each node, connect to top-K most similar nodes (default K=7)
4. **Graph Properties**: Undirected, sparse, no self-loops

### GCN Architecture

**GCNNClassifier** (Default):
- **Layer 1**: Graph Convolutional Layer
  - Input: D_in (1536 for concat, 256 for MLP)
  - Output: 256 hidden channels
  - Normalization: LayerNorm
  - Activation: ReLU
  - Dropout: 0.5
- **Layer 2**: Classification Layer
  - Input: 256
  - Output: 2 (logits for benign/malignant)
  - No activation (raw logits)

**Output**: Class logits (not probabilities). Predictions use argmax on logits.

## Available GCN Models

The following GCN architectures are available (set in `config.yaml`):

- `GCNNClassifier` (default): Standard GCN with layer normalization
- `GCNN_Dot_Product`: GCN with element-wise multiplication
- `GCNN_Concat_Attention`: GCN with attention concatenation
- `GCNN_Prod_Res`: GCN with residual connection
- `GATClassifier`: Graph Attention Network with multi-head attention
- `GraphSAGEClassifier`: GraphSAGE with mean aggregation
- `GNClassifier`: General Graph Neural Network with message passing

## Fine-tuning Options

### Feature Extractor Fine-tuning

Control fine-tuning behavior in `config.yaml`:

```yaml
cnn:
  freeze_backbone: True    # Freeze backbone, train only head
  train_last_n_layers: 1   # Train last N layers + head (when freeze_backbone=True)
  # train_last_n_layers: 0  # Train only head
  # freeze_backbone: False  # Fine-tune entire backbone
```

**Options**:
- `freeze_backbone: False`: Fine-tune entire backbone
- `freeze_backbone: True, train_last_n_layers: 0`: Train only classification head
- `freeze_backbone: True, train_last_n_layers: N`: Train last N layers + head

## Evaluation Metrics

The pipeline computes and saves:

- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: True positives, true negatives, false positives, false negatives
- **Per-fold Results**: Separate metrics for each fold
- **Visualizations**: Confusion matrix plots, loss curves

Results are saved as CSV files in the `artifacts/{backbone}/gcn/` directory.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `config.yaml`
   - Process one backbone at a time using `--backbones`
   - Process one fold at a time using `--folds`

2. **MAE Checkpoint Not Found**
   - Verify path in `config.yaml` is correct
   - Or set `MAE_CKPT` environment variable
   - If checkpoint is not available, skip MAE backbone: `--backbones custom resnet18 dino_vits8 ijepa_vit`

3. **Missing Feature Type Directories**
   - The code will use bmode images as fallback, but this is not recommended
   - Create proper directory structure for all three feature types

4. **PyTorch Geometric Installation Issues**
   - For CPU-only: `pip install torch-geometric`
   - For CUDA: Follow [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### Dependencies Issues

If you encounter import errors:

```bash
# Reinstall core dependencies
pip install --upgrade torch torchvision torch-geometric

# For transformer backbones
pip install --upgrade timm transformers

# For data processing
pip install --upgrade pandas numpy scikit-learn
```

## File Structure

```
MF-GCN/
├── README.md                    # This file
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main pipeline script
├── src/
│   ├── __init__.py
│   ├── cnn_backbones.py         # Backbone model definitions
│   ├── cnn_models.py            # CustomCNN architecture
│   ├── data.py                  # Data loading utilities
│   ├── feature_extraction.py    # Feature extraction training
│   ├── fusion_mlp.py            # MLP fusion implementation
│   ├── gcn_train.py             # GCN training and evaluation
│   ├── graphs.py                # Graph construction
│   ├── metrics.py               # Evaluation metrics and plotting
│   └── multimodal_pa_fusion.py  # Parallel attention fusion (optional)
└── utils/
    ├── __init__.py
    └── GN_models.py             # GCN model architectures
```


## Acknowledgments

- PyTorch Geometric for graph neural network implementations
- timm library for vision transformer models
- HuggingFace Transformers for i-JEPA model

