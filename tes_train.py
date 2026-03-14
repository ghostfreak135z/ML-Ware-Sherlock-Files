import torch
import json
from torch.utils.data import DataLoader
from dataset import VideoDataset

FRAME_DIR = "frames/train"

with open("dataset/train_labels.json") as f:
    labels = json.load(f)

dataset = VideoDataset(FRAME_DIR, labels)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

frames, order = next(iter(loader))

print("frames shape:", frames.shape)
print("order shape:", order.shape)