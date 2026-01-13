# Multi-Scale Convolutional Neural Network (MSCNN)

This repository contains the implementation of a Multi-Scale Convolutional Neural Network (MSCNN) designed for ultrasound-based disease classification. The model is developed to capture diagnostically relevant patterns at multiple spatial resolutions and supports both single-feature and multi-feature ultrasound inputs.

## Overview

Ultrasound images often exhibit structures of interest at varying spatial scales due to differences in lesion size, depth, and acoustic properties. The MSCNN architecture addresses this challenge by explicitly extracting features using parallel convolutional kernels of different sizes within each network block. This design enables robust representation learning across heterogeneous ultrasound imaging conditions.

The MSCNN framework is used for binary classification tasks in breast and liver ultrasound imaging:
- **Breast ultrasound:** benign vs. malignant lesion classification  
- **Liver ultrasound:** non-alcoholic steatohepatitis (NASH) vs. non-alcoholic fatty liver disease (NAFLD)

## Model Architecture

The MSCNN architecture consists of multiple convolutional branches operating at different spatial scales:

- Three multi-scale convolutional blocks per branch
- Parallel convolutional filters with kernel sizes **3×3**, **5×5**, and **7×7**
- Each block includes:
  - Convolution
  - Batch normalization
  - ReLU activation
  - Average pooling

Each multi-scale block outputs a 256-dimensional feature vector. Features from all blocks are concatenated to form a 768-dimensional representation, which is further processed through squeeze and fully connected layers. Final predictions are produced using a Softmax classification layer.

In the multi-feature configuration, identical MSCNN branches are used to process each ultrasound-derived feature representation independently, and their outputs are combined using a late fusion strategy.

## Input Data

The model accepts ultrasound imaging data acquired from clinical breast and liver examinations. Depending on the configuration, inputs include:

- **B-mode ultrasound images**
- **Quality-Improved Ultrasound (QIUS) images**
- **Enhanced Ultrasound (EUS) images**

QIUS and EUS images are generated from the original B-mode data using local phase–based and feature enhancement techniques developed by **PONS**. All feature representations are derived from the same acquisition and remain spatially aligned.

Input images are resized to a fixed spatial resolution and intensity-normalized prior to training. Only ultrasound imaging data are used; no patient-identifiable information or non-imaging clinical data are included.

## Output

The model produces a Softmax-based probability vector corresponding to the predefined classification categories. The predicted class label is determined by selecting the class with the highest probability score. Output probabilities may also be used for confidence estimation and downstream performance evaluation.



Che H, Brown LG, Foran DJ, Nosher JL, Hacihaliloglu I. Liver disease classification from ultrasound using multi-scale CNN. International Journal of Computer Assisted Radiology and Surgery. 2021 Jun 7:1-2.