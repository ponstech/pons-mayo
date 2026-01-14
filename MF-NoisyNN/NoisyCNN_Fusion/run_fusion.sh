#!/bin/bash

# NoisyCNN Fusion Training Script for Breast Cancer Classification
# Usage: bash run_fusion.sh [fusion_type] [fold] [noise_type] [noise_str] [noise_layer]

# Default parameters
FUSION_TYPE=${1:-mid}          # mid or late
FOLD=${2:-1}                    # 1, 2, or 3
NOISE_TYPE=${3:-linear}         # linear, gaussian, or impulse
NOISE_STR=${4:-0.1}             # Noise strength
NOISE_LAYER=${5:-4}             # Layer to inject noise (1-4)
BATCH_SIZE=${6:-32}             # Batch size
EPOCHS=${7:-100}                # Number of epochs
LR=${8:-0.001}                  # Learning rate
GPU_ID=${9:-0}                  # GPU ID
DATASET_PATH=${10:-../combined-breast-3fold}  # Dataset path

echo "=========================================="
echo "NoisyCNN Fusion Training"
echo "=========================================="
echo "Fusion Type: $FUSION_TYPE"
echo "Fold: $FOLD"
echo "Noise Type: $NOISE_TYPE"
echo "Noise Strength: $NOISE_STR"
echo "Noise Layer: $NOISE_LAYER"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "GPU ID: $GPU_ID"
echo "Dataset Path: $DATASET_PATH"
echo "=========================================="

cd NoisyNN-main/NoisyCNN_CircularShiftQ

python Main_Fusion.py \
    --fusion_type $FUSION_TYPE \
    --fold $FOLD \
    --noise_type $NOISE_TYPE \
    --noise_str $NOISE_STR \
    --noisy_layer $NOISE_LAYER \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCHS \
    --lr $LR \
    --gpu_id $GPU_ID \
    --breast_dataset_path $DATASET_PATH \
    --class_num 2 \
    --datasets BreastCancer

echo "Training completed!"

