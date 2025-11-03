import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import requests
import time
from datetime import datetime
import os
from relay_controller import trigger_alert, cleanup_relay  # ✅ Updated import

# ✅ Run headless (no GUI required)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from telegram_config import BOT_TOKEN, CHAT_ID, CONFIDENCE_THRESHOLD
except ImportError:
    BOT_TOKEN = None
    CHAT_ID = None
    CONFIDENCE_THRESHOLD = 0.8
    print("⚠️ Telegram config not found. Update telegram_config.py with your bot details.")


class AnimalDetector:
    def __init__(self, model_path='best_animal_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading model on: {self.device}")

        # Telegram setup
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.last_sent_time = 0
        self.min_interval = 5
        self.detection_start_time = None
        self.current_animal = None
        self.detection_duration = 5

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Classes
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            self.class_names = ['Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
                                'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit',
                                'Skunk', 'Tiger', 'Wild Boar']
            print("⚠️ Using default class names")

        print(f"📋 Classes: {self.class_names}")

        # Build model
        self.model = models.resnet50(weights=None)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'fc.4.weight' in state_dict:
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.class_names))
            )
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        os.makedirs("detections", exist_ok=True)

        # Define wild animals for alert
        self.wild_animals = ["Elephant", "Tiger", "Wild Boar"]

    def send_to_telegram(self, frame, animal, confidence, duration):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections/detection_{animal}_{confidence:.3f}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            message = (
                f"🐾 *Wild Animal Alert!*\n"
                f"🔍 *Species:* {animal}\n"
                f"📊 *Confidence:* {confidence*100:.1f}%\n"
                f"⏱ *Visible for:* {duration:.1f}s\n"
                f"🕰 {datetime.now().strftime('%H:%M:%S')}"
            )

            print("\n" + "=" * 40)
            print(message.replace('*', ''))
            print("=" * 40 + "\n")

            if not self.bot_token or not self.chat_id:
                print("⚠️ Telegram not configured, skipping send")
                return

            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            with open(filename, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': message, 'parse_mode': 'Markdown'}
                requests.post(url, files=files, data=data, timeout=10)
                print(f"📤 Sent {animal} detection to Telegram")

        except Exception as e:
            print(f"❌ Telegram send error: {e}")

    def predict_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam.")
            return

        print("📹 Headless webcam detection started (press Ctrl+C to stop)")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame read failed.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probs, 0)

            animal = self.class_names[predicted.item()]
            conf = confidence.item()
            now = time.time()

            if conf > self.confidence_threshold:
                if self.current_animal == animal:
                    duration = now - self.detection_start_time
                    print(f"⏳ {animal} ({conf * 100:.1f}%) - in frame for {duration:.1f}s")

                    if duration >= self.detection_duration and (now - self.last_sent_time) > self.min_interval:
                        self.last_sent_time = now
                        self.detection_start_time = now
                        self.send_to_telegram(frame, animal, conf, duration)

                        # ✅ Trigger alert for wild animals
                        if animal in self.wild_animals:
                            print(f"🚨 WILD ANIMAL DETECTED: {animal}")
                            trigger_alert(duration=5)
                else:
                    self.current_animal = animal
                    self.detection_start_time = now
            else:
                self.current_animal = None
                self.detection_start_time = None

            time.sleep(0.5)

        cap.release()


if __name__ == "__main__":
    try:
        detector = AnimalDetector()
        print("✅ Model loaded successfully!")
        detector.predict_webcam()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cleanup_relay()
