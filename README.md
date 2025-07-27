# 🎥 Video Anomaly Detection using Spatio-Temporal Autoencoder

This project implements a deep learning pipeline for detecting anomalies in surveillance videos using a **Spatio-Temporal Autoencoder**. The model is trained to reconstruct sequences of normal events, and any significant deviation in reconstruction error is flagged as anomalous.

## 🚀 Project Overview

- **Objective**: Identify unusual or suspicious activity in video surveillance using reconstruction loss.
- **Approach**: Convert video into spatio-temporal cuboids and train an autoencoder to learn regular patterns.
- **Framework**: PyTorch-based implementation with preprocessing in OpenCV and FFmpeg.

---

## 🧠 Model Architecture

- **Encoder**: 3D convolution layers to encode spatial-temporal cuboids.
- **Decoder**: Mirrors the encoder to reconstruct the input sequence.
- **Loss Function**: Mean Squared Error (MSE) for reconstruction loss.
- **Anomaly Detection**: If reconstruction loss > threshold, frame is flagged as anomalous.

---


## 📦 Setup

### 🔧 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- FFmpeg
- Matplotlib
- NumPy

📊 Output
Reconstruction loss is calculated per cuboid.

Frames with loss exceeding the threshold are flagged.

Plots and logs are saved for inspection.

✅ Results
The model was able to detect anomalies with a sharp rise in reconstruction loss.

Normal activities were reconstructed well, while unusual frames showed high deviation.

📌 Future Improvements
Use ConvLSTM or 3D ResNet-based architecture for richer spatio-temporal modeling.

Apply frame-level heatmap overlays for better anomaly localization.

Integrate threshold tuning via validation ROC-AUC.

