Brain Tumor MRI Detection & Classification

An end-to-end Deep Learning web application designed to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans into four distinct categories using Keras, TensorFlow, and Streamlit.

Live Web Application: [Brain Tumor Detector](https://braintumordetector26.streamlit.app/)

---

Project Overview

Brain tumor identification from MRI scans requires high precision and speed. This project implements a Multi-Layer Perceptron (MLP) architecture built with TensorFlow/Keras to analyze MRI scans and provide real-time probabilistic classification for clinical decision support and educational purposes.

Target Classes:
- 🔴 Glioma
- 🟠 Meningioma
- 🔵 Pituitary
- 🟢 No Tumor

---

Tech Stack & Tools

- Language: Python 3.11
- Deep Learning Framework: TensorFlow / Keras
- Image Processing: OpenCV, NumPy, Pillow
- Web Interface: Streamlit
- Version Control & LFS: Git & Git LFS (for large `.keras` model binaries)
- Deployment Platform: Streamlit Community Cloud

---

 Model Architecture & Data Pipeline

1. Preprocessing:
   - Images are resized to 128x128 pixels.
   - Channel order is formatted to BGR color space to maintain exact consistency with the training pipeline.
   - Pixel intensities are normalized to the `[0, 1]` range (`float32`).
2. Architecture:
   - Input Layer: Flatten (49,152 features)
   - Hidden Layers: Fully Connected Dense layers (512 ➔ 256 ➔ 128 units) with ReLU activation.
   - Regularization: Dropout layers (0.3 rate) applied after each hidden layer to prevent overfitting.
   - Output Layer: Dense layer with Softmax activation across 4 classes.

---

 Local Installation & Setup

If you want to run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YusufAbozhw/Brain_Tumor_Detector.git](https://github.com/YusufAbozhw/Brain_Tumor_Detector.git)
   cd Brain_Tumor_Detector