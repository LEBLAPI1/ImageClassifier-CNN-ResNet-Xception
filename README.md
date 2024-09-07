# CIFAR-10 Image Classification using CNN and Transfer Learning

## Overview
This project aims to classify images from the **CIFAR-10** dataset using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. The CIFAR-10 dataset consists of 60,000 low-resolution images (32x32) across 10 categories, such as airplanes, automobiles, birds, cats, and others. The project implements both a CNN-based model from scratch and transfer learning with pre-trained models such as **ResNet50** and **Xception**.

## Problem Statement
The objective is to develop a deep learning model that can accurately classify CIFAR-10 images into 10 distinct classes. Challenges include:
- **High variability** in image content.
- **Small image size** (32x32), which limits the detail available for the model.
- **Efficient use of computational resources** during training.

## Approach
The project employs the following approaches:
- **Custom CNN**: Built from scratch using multiple convolutional layers, batch normalization, dropout, and pooling.
- **Transfer Learning**: Leveraged pre-trained models such as ResNet50 and Xception to benefit from robust feature extraction capabilities.
- **Hyperparameter Tuning**: Applied grid search for optimizing learning rate, batch size, optimizer, and other parameters to maximize model performance.

## Techniques and Tools
- **Deep Learning Framework**: TensorFlow/Keras and PyTorch.
- **Pre-trained Models**: ResNet50 and Xception.
- **Data Augmentation**: Horizontal flipping, random cropping, and normalization.
- **Dimensionality Reduction**: PCA and t-SNE for EDA.
- **Regularization**: Dropout and batch normalization to prevent overfitting.

## Results
- **CNN**: Achieved a test accuracy of **82.5%**.
- **Xception**: Achieved a test accuracy of **74.65%** after tuning.
- **ResNet50**: Lower accuracy, reaching **38.13%**, indicating potential limitations of transfer learning for CIFAR-10 with this model.

## Dataset
The CIFAR-10 dataset consists of:
- **Training set**: 50,000 images across 10 classes.
- **Test set**: 10,000 images across the same 10 classes.
The dataset is publicly available and commonly used for benchmarking in machine learning.

## Key Features
1. **Exploratory Data Analysis (EDA)**: Detailed analysis of the dataset's pixel intensities, class distribution, and edge detection.
2. **Model Architectures**:
   - Custom CNN built from scratch.
   - Transfer learning with ResNet50 and Xception.
3. **Hyperparameter Tuning**: Grid search for optimizing learning rate, batch size, and regularization.
4. **Visualization**:
   - Accuracy and loss curves to monitor training.
   - Dimensionality reduction using PCA and t-SNE.


