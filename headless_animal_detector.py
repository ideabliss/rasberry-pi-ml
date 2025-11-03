import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import requests
import time
from datetime import datetime
import os

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
        self.min_interval = 5  # Minimum 5 seconds between messages

        # Detection persistence tracking
        self.detection_start_time = None
        self.current_animal = None
        self.detection_duration = 5  # Require 5 seconds of detection

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different model file formats
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            self.class_names = ['Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
                                'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit',
                                'Skunk', 'Tiger', 'Wild Boar']
            print("⚠️ Using default class names")

        print(f"📋 Classes: {self.class_names}")

        if 'accuracy' in checkpoint:
            print(f"🎯 Model accuracy: {checkpoint['accuracy']:.4f}")
        else:
            print("📊 Model accuracy: Not available")

        # Create model architecture
        self.model = models.resnet50(weights=None)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Adjust FC layer based on checkpoint
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

        try:
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Trying ResNet34 fallback...")
            self.model = models.resnet34(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded with ResNet34 architecture")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Create output folder for saved detections
        os.makedirs("detections", exist_ok=True)

    def predict_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        return self.class_names[predicted.item()], confidence.item()

    def send_to_telegram(self, frame, animal, confidence):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections/detection_{animal}_{confidence:.3f}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            if not self.bot_token or not self.chat_id:
                print("⚠️ Telegram not configured, skipping send")
                return

            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            with open(filename, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': f"🐾 Animal Detected!\n🔍 Species: {animal}\n📊 Confidence: {confidence:.1%}\n🕰 {datetime.now().strftime('%H:%M:%S')}"
                }
                response = requests.post(url, files=files, data=data, timeout=10)
                if response.status_code == 200:
                    print(f"📤 Sent {animal} detection to Telegram")
                else:
                    print(f"❌ Telegram error: {response.status_code}")

        except Exception as e:
            print(f"❌ Telegram send error: {e}")

    def predict_webcam(self):
        # Try multiple camera sources
        cap = None
        for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            for i in range(3):
                print(f"Trying camera {i} with backend {backend}...")
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    time.sleep(2)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Using camera {i} with backend {backend}")
                        break
                    else:
                        cap.release()
                        cap = None
            if cap and cap.isOpened():
                break

        if not cap or not cap.isOpened():
            print("❌ Could not open webcam.")
            print("💡 Try: sudo modprobe uvcvideo OR connect a USB camera.")
            return

        print("📹 Headless webcam detection started (press Ctrl+C to stop)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)

            animal = self.class_names[predicted.item()]
            conf = confidence.item()
            current_time = time.time()

            # Detection persistence logic
            if conf > self.confidence_threshold:
                if self.current_animal == animal:
                    if self.detection_start_time and (current_time - self.detection_start_time) >= self.detection_duration:
                        if (self.bot_token and self.chat_id) and (current_time - self.last_sent_time) > self.min_interval:
                            self.send_to_telegram(frame, animal, conf)
                            self.last_sent_time = current_time
                            self.detection_start_time = None
                    else:
                        elapsed = current_time - (self.detection_start_time or current_time)
                        remaining = max(0, self.detection_duration - elapsed)
                        print(f"⏳ Detecting {animal} ({conf:.2f}) - {remaining:.1f}s remaining")
                else:
                    self.current_animal = animal
                    self.detection_start_time = current_time
            else:
                self.detection_start_time = None
                self.current_animal = None

            # 🪶 Log detections in console
            print(f"Detected: {animal} ({conf:.2f})")
            time.sleep(0.5)  # small delay to prevent log flooding

        cap.release()
        print("📹 Webcam closed")


def test_image_prediction(detector, image_path):
    try:
        animal, confidence = detector.predict_image(image_path)
        print(f"🔍 Prediction: {animal}")
        print(f"📊 Confidence: {confidence:.4f}")
        return animal, confidence
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None


if __name__ == "__main__":
    try:
        detector = AnimalDetector()
        print("✅ Model loaded successfully!")

        print("\n🎥 Starting headless webcam detection...")
        if detector.bot_token and detector.chat_id:
            print(f"🤖 Telegram bot enabled - will send photos when confidence > {detector.confidence_threshold:.0%}")
        else:
            print("⚠️ Telegram not configured - update telegram_config.py")

        detector.predict_webcam()

    except FileNotFoundError:
        print("❌ Model file 'best_animal_model.pth' not found!")
        print("🔧 Train your model first.")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
