import cv2
import os
from tqdm import tqdm

VIDEO_DIR = r"D:\Machine Learning\ML WARE\dataset\train"
SAVE_DIR = r"D:\Machine Learning\ML WARE\frames\train"
MAX_FRAMES = 64

os.makedirs(SAVE_DIR, exist_ok=True)

videos = os.listdir(VIDEO_DIR)

for v in tqdm(videos):

    path = os.path.join(VIDEO_DIR, v)
    cap = cv2.VideoCapture(path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // MAX_FRAMES, 1)

    folder = os.path.join(SAVE_DIR, v.split(".")[0])
    os.makedirs(folder, exist_ok=True)

    frame_id = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            frame = cv2.resize(frame,(224,224))
            cv2.imwrite(f"{folder}/{count}.jpg", frame)
            count += 1

        frame_id += 1

    cap.release()