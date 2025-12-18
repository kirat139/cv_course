import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)
time.sleep(2)  # warm up camera

background = None

for i in range(60):
    ret, frame = cap.read()
    if not ret:
        continue
    # Flip so it behaves like a mirror
    frame = cv2.flip(frame, 1)
    background = frame

# background now holds a clean background frame
