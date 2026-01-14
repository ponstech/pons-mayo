#!/usr/bin/env python3
"""
Example running scripts for NoisyCNN Fusion
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

# Example 1: Mid Fusion with Linear Noise
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "mid",
    "--fold", "1",
    "--noise_type", "linear",
    "--noise_str", "0.1",
    "--noisy_layer", "4",
    "--batch_size", "32",
    "--epoch", "100",
    "--lr", "0.001",
    "--gpu_id", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--class_num", "2",
    "--datasets", "BreastCancer"
], "Mid Fusion, Fold 1, Linear Noise, Layer 4")

# Example 2: Late Fusion with Gaussian Noise
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "late",
    "--fold", "1",
    "--noise_type", "gaussian",
    "--noise_str", "0.1",
    "--noisy_layer", "2",
    "--batch_size", "32",
    "--epoch", "100",
    "--lr", "0.001",
    "--gpu_id", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--class_num", "2",
    "--datasets", "BreastCancer",
    "--gau_mean", "0.0",
    "--gau_var", "1.0"
], "Late Fusion, Fold 1, Gaussian Noise, Layer 2")

# Example 3: Mid Fusion with Impulse Noise
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "mid",
    "--fold", "2",
    "--noise_type", "impulse",
    "--noise_str", "0.15",
    "--noisy_layer", "3",
    "--batch_size", "32",
    "--epoch", "100",
    "--lr", "0.001",
    "--gpu_id", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--class_num", "2",
    "--datasets", "BreastCancer"
], "Mid Fusion, Fold 2, Impulse Noise, Layer 3")

# Example 4: Late Fusion with Linear Noise, Different Layer
run_command([
    "python", "Main_Fusion.py",
    "--fusion_type", "late",
    "--fold", "3",
    "--noise_type", "linear",
    "--noise_str", "0.05",
    "--noisy_layer", "1",
    "--batch_size", "32",
    "--epoch", "100",
    "--lr", "0.001",
    "--gpu_id", "0",
    "--breast_dataset_path", "../combined-breast-3fold",
    "--class_num", "2",
    "--datasets", "BreastCancer"
], "Late Fusion, Fold 3, Linear Noise, Layer 1")

print("\n" + "="*60)
print("All example commands listed above.")
print("Uncomment the subprocess.run() line to execute commands.")
print("="*60)

