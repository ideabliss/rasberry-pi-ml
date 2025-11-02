import cv2

# Test different camera indices
for i in range(3):
    print(f"Testing camera {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera {i} working - Frame shape: {frame.shape}")
            cv2.imshow(f'Camera {i} Test', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        else:
            print(f"❌ Camera {i} opened but no frame")
    else:
        print(f"❌ Camera {i} failed to open")
    
    cap.release()

print("Camera test complete")