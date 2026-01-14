#!/bin/bash

# NoisyViT Fusion Training Script for Breast Cancer Classification
# Usage: bash run_fusion.sh [fusion_type] [fold] [noise_type] [strength] [layer] [scale] [res] [patch_size]

# Default parameters
FUSION_TYPE=${1:-mid}          # mid or late
FOLD=${2:-1}                    # 1, 2, or 3
NOISE_TYPE=${3:-linear}        # linear, gaussian, or impulse
STRENGTH=${4:-0.1}             # Noise strength
LAYER=${5:-11}                 # Layer to inject noise
SCALE=${6:-tiny}               # Model scale: tiny, small, base, large, huge
RES=${7:-224}                  # Image resolution
PATCH_SIZE=${8:-16}            # Patch size
BATCH_SIZE=${9:-10}            # Batch size
EPOCHS=${10:-100}              # Number of epochs
LR=${11:-0.0001}               # Learning rate
GPU_ID=${12:-0}                # GPU ID
OPTIMALQ=${13:-0}              # Use OptimalQ (1) or linear (0)
DATASET_PATH=${14:-../combined-breast-3fold}  # Dataset path

echo "=========================================="
echo "NoisyViT Fusion Training"
echo "=========================================="
echo "Fusion Type: $FUSION_TYPE"
echo "Fold: $FOLD"
echo "Noise Type: $NOISE_TYPE"
echo "Strength: $STRENGTH"
echo "Layer: $LAYER"
echo "Scale: $SCALE"
echo "Resolution: $RES"
echo "Patch Size: $PATCH_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "GPU ID: $GPU_ID"
echo "OptimalQ: $OPTIMALQ"
echo "Dataset Path: $DATASET_PATH"
echo "=========================================="

cd NoisyNN-main/NoisyViT-OptimalQ

python Main_Fusion.py \
    --fusion_type $FUSION_TYPE \
    --fold $FOLD \
    --noise_type $NOISE_TYPE \
    --strength $STRENGTH \
    --layer $LAYER \
    --scale $SCALE \
    --res $RES \
    --patch_size $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --te_batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --gpu_id $GPU_ID \
    --OptimalQ $OPTIMALQ \
    --breast_dataset_path $DATASET_PATH \
    --num_classes 2 \
    --datasets BreastCancer

echo "Training completed!"

