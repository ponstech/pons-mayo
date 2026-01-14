#!/bin/bash

# Run all experiments for NoisyCNN Fusion
# This script runs experiments for all combinations of fusion types, folds, and noise types

DATASET_PATH=${1:-../combined-breast-3fold}
GPU_ID=${2:-0}

echo "=========================================="
echo "Running All NoisyCNN Fusion Experiments"
echo "=========================================="

# Fusion types
FUSION_TYPES=("mid" "late")

# Folds
FOLDS=(1 2 3)

# Noise types
NOISE_TYPES=("linear" "gaussian" "impulse")

# Noise layers
NOISE_LAYERS=(1 2 3 4)

# Noise strengths
NOISE_STRS=(0.05 0.1 0.15)

for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
            for NOISE_LAYER in "${NOISE_LAYERS[@]}"; do
                for NOISE_STR in "${NOISE_STRS[@]}"; do
                    echo ""
                    echo "=========================================="
                    echo "Running: $FUSION_TYPE, Fold $FOLD, $NOISE_TYPE, Layer $NOISE_LAYER, Str $NOISE_STR"
                    echo "=========================================="
                    
                    cd NoisyNN-main/NoisyCNN_CircularShiftQ
                    
                    python Main_Fusion.py \
                        --fusion_type $FUSION_TYPE \
                        --fold $FOLD \
                        --noise_type $NOISE_TYPE \
                        --noise_str $NOISE_STR \
                        --noisy_layer $NOISE_LAYER \
                        --batch_size 32 \
                        --epoch 100 \
                        --lr 0.001 \
                        --gpu_id $GPU_ID \
                        --breast_dataset_path $DATASET_PATH \
                        --class_num 2 \
                        --datasets BreastCancer
                    
                    cd ../..
                    
                    echo "Completed: $FUSION_TYPE, Fold $FOLD, $NOISE_TYPE, Layer $NOISE_LAYER, Str $NOISE_STR"
                    sleep 5  # Wait 5 seconds between experiments
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

