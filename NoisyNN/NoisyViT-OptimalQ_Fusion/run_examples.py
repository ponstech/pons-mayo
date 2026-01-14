#!/usr/bin/env python3
"""
Example running scripts for NoisyViT Fusion
This script provides example commands for running different experiments
"""

import subprocess
import sys
import os

# Change to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def run_command(cmd, description):
    """Run a command and print description"""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # Uncomment the line below to actually run the command
    # subprocess.run(cmd, check=True)
    print("(Command commented out - uncomment to run)")

# Example 1: Mid Fusion with Linear Noise, Tiny Scale
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "mid",
    "--fold", "1",
    "--noise_type", "linear",
    "--strength", "0.1",
    "--layer", "11",
    "--scale", "tiny",
    "--res", "224",
    "--patch_size", "16",
    "--batch_size", "10",
    "--te_batch_size", "10",
    "--epochs", "100",
    "--lr", "0.0001",
    "--gpu_id", "0",
    "--OptimalQ", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--num_classes", "2",
    "--datasets", "BreastCancer"
], "Mid Fusion, Fold 1, Linear Noise, Tiny Scale, Layer 11")

# Example 2: Late Fusion with OptimalQ
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "late",
    "--fold", "1",
    "--noise_type", "linear",
    "--strength", "0.1",
    "--layer", "11",
    "--scale", "tiny",
    "--res", "224",
    "--patch_size", "16",
    "--batch_size", "10",
    "--te_batch_size", "10",
    "--epochs", "100",
    "--lr", "0.0001",
    "--gpu_id", "0",
    "--OptimalQ", "1",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--num_classes", "2",
    "--datasets", "BreastCancer"
], "Late Fusion, Fold 1, OptimalQ, Tiny Scale, Layer 11")

# Example 3: Mid Fusion with Gaussian Noise
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "mid",
    "--fold", "2",
    "--noise_type", "gaussian",
    "--strength", "0.1",
    "--layer", "9",
    "--scale", "small",
    "--res", "224",
    "--patch_size", "16",
    "--batch_size", "10",
    "--te_batch_size", "10",
    "--epochs", "100",
    "--lr", "0.0001",
    "--gpu_id", "0",
    "--OptimalQ", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--num_classes", "2",
    "--datasets", "BreastCancer",
    "--gau_mean", "0.0",
    "--gau_var", "1.0"
], "Mid Fusion, Fold 2, Gaussian Noise, Small Scale, Layer 9")

# Example 4: Late Fusion with Impulse Noise
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "late",
    "--fold", "3",
    "--noise_type", "impulse",
    "--strength", "0.15",
    "--layer", "6",
    "--scale", "tiny",
    "--res", "224",
    "--patch_size", "16",
    "--batch_size", "10",
    "--te_batch_size", "10",
    "--epochs", "100",
    "--lr", "0.0001",
    "--gpu_id", "0",
    "--OptimalQ", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--num_classes", "2",
    "--datasets", "BreastCancer"
], "Late Fusion, Fold 3, Impulse Noise, Tiny Scale, Layer 6")

# Example 5: Mid Fusion with Base Scale
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "mid",
    "--fold", "1",
    "--noise_type", "linear",
    "--strength", "0.1",
    "--layer", "11",
    "--scale", "base",
    "--res", "224",
    "--patch_size", "16",
    "--batch_size", "8",
    "--te_batch_size", "8",
    "--epochs", "100",
    "--lr", "0.0001",
    "--gpu_id", "0",
    "--OptimalQ", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--num_classes", "2",
    "--datasets", "BreastCancer"
], "Mid Fusion, Fold 1, Linear Noise, Base Scale, Layer 11")

print("\n" + "="*60)
print("All example commands listed above.")
print("Uncomment the subprocess.run() line to execute commands.")
print("="*60)

