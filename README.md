# ML-Ware-Sherlock-Files

# Temporal Frame Ordering in Videos

This project focuses on reconstructing the correct chronological order of shuffled video frames using deep learning techniques.

The system processes video frames, extracts visual features, models temporal relationships, and predicts the correct ordering of frames.

---

## Project Overview

In many real-world scenarios, videos may contain frames that are shuffled or unordered.  
The goal of this project is to automatically predict the correct temporal order of frames.

We solve this problem using a deep learning pipeline that combines:

- Visual feature extraction
- Temporal modeling
- Frame ranking prediction

---

## Pipeline

The overall pipeline of the system is:

Video → Frame Extraction → Feature Extraction → Transformer → Frame Ranking → Frame Order

### Steps

1. Extract frames from videos.
2. Resize frames to a fixed resolution.
3. Extract visual features using a pretrained CNN.
4. Model temporal relationships using a Transformer.
5. Predict frame ordering scores.
6. Sort scores to obtain the predicted frame sequence.

---

## Model Architecture

The model consists of the following components.

### Feature Extractor

A pretrained **ResNet50** network extracts high-level visual features from each frame.

Frame → ResNet50 → Feature Vector (2048-dim)

### Temporal Modeling

A **Transformer Encoder** processes the sequence of frame embeddings to learn temporal relationships.

Frame Features → Transformer Encoder → Temporal Representation

### Ranking Head

A linear layer predicts a score for each frame.  
Sorting these scores produces the predicted frame order.

---

## Dataset

The dataset used in this project contains shuffled videos.

| Property | Value |
|--------|--------|
| Training videos | 5611 |
| Test videos | 296 |
| Frame range | 15–300 |
| Average frames | ~100 |

Frames are resized to **224 × 224** before being fed to the model.

---

## Training Setup

| Parameter | Value |
|-----------|------|
| GPU | Tesla P100 |
| Batch Size | 2 |
| Frames per Video | 16 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 3 |

Loss function used during training:

Mean Squared Error (MSE)

---

## Results

During training, the loss decreases significantly, indicating that the model learns meaningful temporal relationships between frames.

Example loss progression:

| Epoch | Loss |
|------|------|
| 0 | ~0.37 |
| 1 | ~0.25 |
| 2 | ~0.20 |

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/video-frame-ordering.git

Install required dependencies:

pip install torch torchvision opencv-python numpy tqdm pandas

---

## Usage

### Extract Frames

python extract_frames.py

### Train the Model

python train.py

---

## Project Structure

video-frame-ordering
│
├── dataset
│   ├── train
│   └── train_labels.json
│
├── frames
│
├── extract_frames.py
├── dataset.py
├── model.py
├── train.py
└── README.md

---

## Future Improvements

Potential improvements include:

- Using **DINOv2** for stronger feature extraction
- Adding **Optical Flow** for motion understanding
- Using **pairwise ranking loss**
- Increasing Transformer depth

---

## Author

Your Name
