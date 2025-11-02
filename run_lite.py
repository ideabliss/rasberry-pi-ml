import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import requests
import time
import os
import csv
from datetime import datetime

# ================================================================
# ✅ Lightweight Headless Animal Detector for Raspberry Pi
# ================================================================
class LiteAnimalDetector:
    def __init__(self, model_path='best_animal_model.pth'):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

        self.class_names = [
            'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
            'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit',
            'Skunk', 'Tiger', 'Wild Boar'
        ]

        # --- Load model ---
        self.model = None
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = nn.Linear(
                self.model.classifier[3].in_features, len(self.class_names)
            )

            try:
                self.model.load_state_dict(
                    checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                )
            except Exception:
                print("⚠️ Checkpoint mismatch → using ResNet34 fallback")
                self.model = models.resnet34(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
                self.model.load_state_dict(
                    checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                )

            self.model.eval()
            print("✅ Model loaded successfully on CPU")

        except Exception as e:
            print(f"❌ Model load error: {e}")
            return

        # --- Telegram Configuration ---
        try:
            from telegram_config import BOT_TOKEN, CHAT_ID, CONFIDENCE_THRESHOLD
            self.bot_token = BOT_TOKEN
            self.chat_id = CHAT_ID
            self.confidence_threshold = CONFIDENCE_THRESHOLD
            print("📲 Telegram alerts enabled")
        except ImportError:
            print("⚠️ No telegram_config.py found — alerts disabled")
            self.bot_token = None
            self.chat_id = None
            self.confidence_threshold = 0.9

        self.last_sent_time = 0

        # --- Image transforms ---
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # --- Create directories ---
        self.output_dir = "detections"
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_file = os.path.join(self.output_dir, "detections_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Animal", "Confidence", "Filepath"])

    # ================================================================
    # 🧩 Main detection loop (headless-friendly)
    # ================================================================
    def predict_webcam(self):
        if not self.model:
            print("❌ Model not loaded, aborting.")
            return

        cap = None
        for cam_id in [0, 1, 2]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                print(f"📹 Camera detected at index {cam_id}")
                break
            cap.release()

        if not cap or not cap.isOpened():
            print("❌ No webcam found. Please connect and retry.")
            return

        # Optimize for Raspberry Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("✅ Webcam ready — detection started (Ctrl + C to stop)")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Frame capture failed.")
                    continue

                frame_count += 1
                if frame_count % 5 != 0:
                    continue  # skip frames to reduce load

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(frame_rgb).unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, pred = torch.max(probs, 0)

                animal = self.class_names[pred.item()]
                confidence = conf.item()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"[{timestamp}] Detected: {animal} ({confidence:.1%})")

                # --- Save detection locally ---
                if confidence > 0.7:
                    filename = os.path.join(
                        self.output_dir,
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{animal}_{confidence:.2f}.jpg"
                    )
                    cv2.imwrite(filename, frame)
                    self.log_detection(timestamp, animal, confidence, filename)

                # --- Telegram alert ---
                if confidence > self.confidence_threshold and self.bot_token:
                    now = time.time()
                    if now - self.last_sent_time > 10:
                        self.send_to_telegram(frame, animal, confidence)
                        self.last_sent_time = now

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            print("📦 Camera released. Exiting...")

    # ================================================================
    # 🧾 Log detection
    # ================================================================
    def log_detection(self, timestamp, animal, confidence, filepath):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, animal, f"{confidence:.3f}", filepath])

    # ================================================================
    # 📲 Send detection to Telegram
    # ================================================================
    def send_to_telegram(self, frame, animal, confidence):
        try:
            filename = f"/tmp/{animal}_{confidence:.2f}.jpg"
            cv2.imwrite(filename, frame)
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            with open(filename, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": f"🐾 {animal} detected!\nConfidence: {confidence:.1%}"
                }
                requests.post(url, files=files, data=data, timeout=5)
            os.remove(filename)
            print(f"📱 Telegram alert sent for {animal}")
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")


# ================================================================
# 🚀 Entry Point
# ================================================================
if __name__ == "__main__":
    detector = LiteAnimalDetector()
    detector.predict_webcam()
