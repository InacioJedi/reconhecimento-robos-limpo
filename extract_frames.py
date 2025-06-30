import cv2
import os

# Pastas e taxa de extração
VIDEO_DIR = 'data/videos'
OUTPUT_DIR = 'data/images'
FPS_EXTRACT = 1  # quadros por segundo

os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'val'), exist_ok=True)

for video_name in os.listdir(VIDEO_DIR):
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps / FPS_EXTRACT))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            # 80% para treino, 20% para val
            dest = 'train' if (saved % 5) < 4 else 'val'
            filename = f"{os.path.splitext(video_name)[0]}_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, dest, filename), frame)
            saved += 1
        count += 1
    cap.release()