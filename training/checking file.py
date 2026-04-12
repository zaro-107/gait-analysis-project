import os
import cv2

folder = r"C:\Users\ashok\OneDrive\Desktop\gait\video_data\waddling"

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"{f} | open={opened} | fps={fps} | frames={frames} | size={w}x{h}")