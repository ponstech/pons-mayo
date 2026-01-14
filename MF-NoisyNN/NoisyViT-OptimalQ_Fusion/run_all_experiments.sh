#!/bin/bash

# Run all experiments for NoisyViT Fusion
# This script runs experiments for all combinations of fusion types, folds, and noise types

DATASET_PATH=${1:-../combined-breast-3fold}
GPU_ID=${2:-0}

echo "=========================================="
echo "Running All NoisyViT Fusion Experiments"
echo "=========================================="

# Fusion types
FUSION_TYPES=("mid" "late")

# Folds
FOLDS=(1 2 3)

# Noise types
NOISE_TYPES=("linear" "gaussian" "impulse")

# Layers (for ViT, typically 0-11 for 12-layer model)
LAYERS=(6 9 11)

# Strengths
STRENGTHS=(0.05 0.1 0.15)

# Model scales
SCALES=("tiny" "small")

for FUSION_TYPE in "${FUSION_TYPES[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
            for LAYER in "${LAYERS[@]}"; do
                for STRENGTH in "${STRENGTHS[@]}"; do
                    for SCALE in "${SCALES[@]}"; do
                        echo ""
                        echo "=========================================="
                        echo "Running: $FUSION_TYPE, Fold $FOLD, $NOISE_TYPE, Layer $LAYER, Str $STRENGTH, Scale $SCALE"
                        echo "=========================================="
                        
                        cd NoisyNN-main/NoisyViT-OptimalQ
                        
                        python Main_Fusion.py \
                            --fusion_type $FUSION_TYPE \
                            --fold $FOLD \
                            --noise_type $NOISE_TYPE \
                            --strength $STRENGTH \
                            --layer $LAYER \
                            --scale $SCALE \
                            --res 224 \
                            --patch_size 16 \
                            --batch_size 10 \
                            --te_batch_size 10 \
                            --epochs 100 \
                            --lr 0.0001 \
                            --gpu_id $GPU_ID \
                            --OptimalQ 0 \
                            --breast_dataset_path $DATASET_PATH \
                            --num_classes 2 \
                            --datasets BreastCancer
                        
                        cd ../..
                        
                        echo "Completed: $FUSION_TYPE, Fold $FOLD, $NOISE_TYPE, Layer $LAYER, Str $STRENGTH, Scale $SCALE"
                        sleep 5  # Wait 5 seconds between experiments
                    done
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

