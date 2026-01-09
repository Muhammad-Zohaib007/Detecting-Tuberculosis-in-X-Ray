# 🫁 Tuberculosis Detection in Chest X-Rays using Deep Learning
![TB Detection Banner](https://cdn.who.int/media/images/default-source/products/global-reports/tb-report/2024/gtb_report-web-banner-2024.tmb-1920v.png?sfvrsn=d47b797e_1)

> 💡 **Did you know?** In 2023, tuberculosis caused approximately **1.25 million deaths** worldwide, making it one of the deadliest infectious diseases globally. Early detection through chest X-rays can significantly improve treatment outcomes. [Source: WHO Global Tuberculosis Report 2024](https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/tb-reports/global-tuberculosis-report-2024/tb-disease-burden/1-2-tb-mortality)

[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org)

## 📌 Project Overview

This project implements a deep learning pipeline for tuberculosis detection in chest X-ray images using a **limited dataset of only 300 images**. Despite the small dataset size, our model achieves **90% accuracy** through innovative training techniques that prevent overfitting and maximize performance.

## 🧪 The Challenge

The primary challenge is developing a robust TB detection model with **only 300 training images** (150 healthy + 150 TB) while avoiding overfitting:

- 📉 **Extremely limited data** - standard deep learning models typically require thousands of examples
- ⚖️ **Balanced classes** but insufficient for conventional deep learning approaches
- 🎯 **Goal**: Achieve high accuracy without overfitting
- 📏 **Constraint**: Model must generalize well to unseen data

This mirrors real-world medical AI challenges where labeled data is scarce, expensive to acquire, and requires expert annotation. The project demonstrates how to effectively train a model with minimal data while maintaining high performance.

## 📊 Dataset Information

The project uses a subset from the Sakha-TB dataset:

| Category | Healthy | TB | Total |
|----------|---------|------|-------|
| **Training** | 150 | 150 | 300 |
| **Testing** | 50 | 50 | 100 |
| **Total** | 200 | 200 | 400 |

## 🧠 Model Architecture

### Core Architecture
- **Base Model**: EfficientNet-B3 (pre-trained on ImageNet)
- **Total Parameters**: 11,485,226
- **Trainable Parameters**: 11,485,226
- **Input Size**: 300×300 pixels
- **Output**: Binary classification (Healthy vs. TB)

### Two-Phase Training Strategy
1. **Phase 1**: Train only the classification head with frozen backbone (10 epochs)
2. **Phase 2**: Fine-tune the entire network with reduced learning rate (50 epochs with early stopping)

### Advanced Techniques
- **Mixup Augmentation**: Creates synthetic training examples by combining pairs of images
- **Learning Rate Scheduling**: Gradual reduction with warm restarts<img width="659" height="590" alt="output1" src="https://github.com/user-attachments/assets/1d99bab2-a020-4c38-abef-06f233dffa2f" />

- **Test-Time Augmentation (TTA)**: Multiple predictions with different augmentations
- **Early Stopping**: Monitors validation performance to prevent overfitting

## 📈 Training Approach

python
def train_model():
    Two-phase training strategy to combat overfitting with limited data
    
    # Phase 1: Train classification head (frozen backbone)
    freeze_backbone(model)
    train(epochs=10, lr=9.76e-4)
    
    # Phase 2: Fine-tune entire network with reduced learning rate
    unfreeze_layers(model, unfreeze_ratio=0.3)
    train(epochs=50, lr=1e-4, early_stopping=True)
    
    # Test-Time Augmentation for inference
    return apply_tta(model)

# Clone the repository
git clone https://github.com/yourusername/tb-detection.git
cd tb-detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
