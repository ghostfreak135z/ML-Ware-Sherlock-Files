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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(3):

    total_loss = 0

    for i, (frames, order) in enumerate(loader):

        frames = frames.to(device).float()
        order = order.to(device).float()

        scores = model(frames)

        loss = criterion(scores, order)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0:
            print("batch", i, "loss", loss.item())


    print("epoch:", epoch, "loss:", total_loss/len(loader))