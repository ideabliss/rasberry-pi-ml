import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"Frame {i}: {frame.shape}, Min: {frame.min()}, Max: {frame.max()}")
        if frame.max() > 0:
            cv2.imshow('Debug', frame)
            cv2.waitKey(1000)
            break
    else:
        print(f"Frame {i}: Failed to read")

cap.release()
cv2.destroyAllWindows()