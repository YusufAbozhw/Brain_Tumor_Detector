# Brain Tumor MRI Detection & Classification

An end-to-end Deep Learning web application designed to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans into four distinct categories using Keras, TensorFlow, and Streamlit.

Live Web Application: [Brain Tumor Detector](https://braintumordetector26.streamlit.app/)

---

## Project Overview

Brain tumor identification from MRI scans requires high precision and speed. This project implements an optimized Multi-Layer Perceptron (MLP) architecture built with TensorFlow/Keras to analyze MRI scans and provide real-time probabilistic classification for clinical decision support and educational purposes.

Note: The web application is deployed using the updated **`model_v2.keras`** weights for improved performance and structure consistency.

Target Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Tech Stack & Tools

- Language: Python 3.11
- Deep Learning Framework: TensorFlow / Keras (Active Model: `Src/model_v2.keras`)
- Image Processing: OpenCV, NumPy, Pillow
- Web Interface: Streamlit
- Version Control & LFS: Git & Git LFS (for large `.keras` model binaries)
- Deployment Platform: Streamlit Community Cloud

---

## Model Architecture & Data Pipeline

1. Preprocessing:
   - Images are resized to 128x128 pixels.
   - Channel order is formatted to BGR color space to maintain exact consistency with the training pipeline.
   - Pixel intensities are normalized to the [0, 1] range (float32).
2. Architecture (`model_v2.keras`):
   - Input Layer: Explicit Keras Input layer followed by Flatten (49,152 features).
   - Hidden Layers: Fully Connected Dense layers (512 -> 256 -> 128 units) with ReLU activation.
   - Regularization: Dropout layers applied to prevent overfitting during training.
   - Output Layer: Dense layer with Softmax activation across 4 classes.

---

## System & Environment Requirements

> Note: This notebook (Src/mlp_project.ipynb) is optimized to run on Kaggle with GPU Acceleration enabled (GPU T4 x2 or P100).

- Hardware Accelerator: Make sure to set the Kaggle Accelerator to GPU (Settings -> Accelerator -> GPU T4 x2).
- Colab Notice: Running this notebook on Google Colab (Free Tier) is not recommended, as it will likely hit RAM limits (OOM crashes) during heavy dataset preprocessing and model training.

---

## Local & Colab Setup Guide

If you want to run this project locally or test it on Google Colab, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YusufAbozhw/Brain_Tumor_Detector.git](https://github.com/YusufAbozhw/Brain_Tumor_Detector.git)
   cd Brain_Tumor_Detector

---

## Disclaimer

This system is developed strictly for educational and demonstration purposes. It is not intended to provide medical advice, diagnosis, or treatment, and should never replace professional medical evaluation.
