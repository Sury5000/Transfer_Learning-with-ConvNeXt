# Transfer Learning with ConvNeXt on Flowers102

This project demonstrates a complete transfer learning pipeline using a pretrained ConvNeXt-Base model on the Flowers102 dataset. The goal is to adapt a large ImageNet-trained model to a small dataset while preventing overfitting using freezing, data augmentation, scheduling, and early stopping.

---

## Project Overview

This work is divided into two major parts:

1. Understanding residual learning through a custom ResNet-style implementation.
2. Applying transfer learning using a pretrained ConvNeXt model.

---

# Part 1 – Residual Network Implementation (Conceptual Foundation)

Before applying transfer learning, a ResNet34-style architecture was implemented from scratch to understand residual connections.

## ResidualUnit Design

Each residual unit contains:

- Two 3×3 convolution layers
- Batch Normalization
- ReLU activation
- Skip connection
- Downsampling using stride when channel size increases

The output is computed as:

    Output = ReLU(Main Path + Skip Connection)

This helps deep networks learn residual mappings and improves gradient flow.

## ResNet34 Architecture

The network structure:

- Initial 7×7 convolution
- Max pooling
- Residual blocks arranged as:
  - 64 filters × 3 blocks
  - 128 filters × 4 blocks
  - 256 filters × 6 blocks
  - 512 filters × 3 blocks
- Adaptive Average Pooling
- Final Linear classification layer

This implementation was used to understand how deep residual networks are structured.

---

# Part 2 – Transfer Learning with ConvNeXt

## Pretrained Model

- Architecture: ConvNeXt-Base
- Pretrained on: ImageNet-1K
- Used pretrained weights

The pretrained lower layers are used as feature extractors.

---

# Dataset – Flowers102

- 102 flower categories
- Very small dataset (approximately 10 training images per class)
- Requires transfer learning due to limited data

Dataset loaded using torchvision.

---

# Modifying the Classifier

The original ConvNeXt model outputs 1000 classes (ImageNet).

The classifier head was modified:

- Original output: 1000 classes
- New output: 102 classes

The final linear layer was replaced to match the Flowers102 dataset.

---

# Freezing Strategy

To prevent overfitting:

- All pretrained layers were frozen
- Only the classifier parameters were unfrozen and trained

This ensures that:

- Learned ImageNet features remain intact
- Only the final layer adapts to the new dataset

---

# Data Augmentation

Strong augmentation techniques were applied:

- Random Horizontal Flip
- Random Rotation (30 degrees)
- Random Resized Crop (224×224)
- Color Jitter
- Normalization using ImageNet statistics

This helps simulate variability and improve generalization.

---

# Optimization Setup

- Loss Function: CrossEntropyLoss
- Optimizer: Adam (training only classifier parameters)
- Learning Rate: 1e-3

---

# Learning Rate Scheduling

ReduceLROnPlateau scheduler was used:

- Monitors validation loss
- Reduces learning rate when validation stops improving
- Factor: 0.5
- Patience: 2 epochs

This helps improve convergence stability.

---

# Early Stopping

A custom EarlyStopping class was implemented:

- Patience: 5 epochs
- Stops training if validation loss does not improve
- Saves the best model weights

This prevents overfitting and unnecessary training.

---

# Training Pipeline

For each epoch:

1. Train on training dataset
2. Validate on validation dataset
3. Update learning rate using scheduler
4. Save best model based on validation loss
5. Check early stopping condition

Maximum training: 30 epochs  
Training may stop earlier if validation stagnates.

---

# Key Learnings

- Transfer learning is highly effective for small datasets.
- Freezing pretrained layers prevents overfitting.
- Strong augmentation improves robustness.
- Adaptive learning rate scheduling improves optimization.
- Early stopping controls training dynamics.
- Classifier head replacement enables adaptation to new tasks.

---

# Conclusion

This project demonstrates a complete transfer learning workflow:

Residual Network Understanding → Pretrained ConvNeXt → Classifier Replacement → Freezing → Augmentation → Scheduler → Early Stopping.

The final pipeline successfully adapts a large pretrained model to a small dataset while maintaining generalization and stability.
