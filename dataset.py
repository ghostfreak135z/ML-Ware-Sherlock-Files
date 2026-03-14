import os
import torch
import cv2
from torch.utils.data import Dataset

MAX_FRAMES = 16

class VideoDataset(Dataset):

    def __init__(self, frame_dir, labels):
        self.frame_dir = frame_dir
        self.labels = labels
        self.videos = list(labels.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        vid = self.videos[idx]
        path = os.path.join(self.frame_dir, vid)

        frames = sorted(os.listdir(path))[:MAX_FRAMES]

        imgs = []

        for f in frames:
            img = cv2.imread(os.path.join(path, f))
            img = torch.tensor(img).permute(2,0,1).float()/255
            imgs.append(img)

        while len(imgs) < MAX_FRAMES:
            imgs.append(imgs[-1])

        imgs = torch.stack(imgs)

        #order = torch.tensor([x-1 for x in self.labels[vid]])[:MAX_FRAMES]
        order = [x-1 for x in self.labels[vid]]

        if len(order) > MAX_FRAMES:
            order = order[:MAX_FRAMES]

        while len(order) < MAX_FRAMES:
            order.append(order[-1])

        order = torch.tensor(order)
        return imgs, order