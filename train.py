import torch
import json
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import FrameOrderModel

FRAME_DIR = "frames/train"

with open("dataset/train_labels.json") as f:
    labels = json.load(f)

dataset = VideoDataset(FRAME_DIR, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

device = "cuda"

model = FrameOrderModel().to(device)

frames, order = next(iter(loader))

frames = frames.to(device).float()

scores = model(frames)

print("scores shape:", scores.shape)