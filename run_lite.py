import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import requests
import time
import os
from datetime import datetime

class LiteAnimalDetector:
    def __init__(self, model_path='best_animal_model.pth'):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')

        self.class_names = [
            'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
            'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit',
            'Skunk', 'Tiger', 'Wild Boar'
        ]

        # ✅ Define transform early so it's always available
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # ✅ Telegram config (optional)
        try:
            from telegram_config import BOT_TOKEN, CHAT_ID, CONFIDENCE_THRESHOLD
            self.bot_token = BOT_TOKEN
            self.chat_id = CHAT_ID
            self.confidence_threshold = CONFIDENCE_THRESHOLD
        except ImportError:
            print("⚠️ No telegram_config.py found — alerts disabled")
            self.bot_token = None
            self.chat_id = None
            self.confidence_threshold = 0.9

        self.last_sent_time = 0
        self.output_dir = "detections"
        os.makedirs(self.output_dir, exist_ok=True)

        # ✅ Load lightweight model safely
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            try:
                self.model = models.mobilenet_v3_small(weights=None)
                self.model.classifier[3] = nn.Linear(
                    self.model.classifier[3].in_features, len(self.class_names)
                )
                self.model.load_state_dict(
                    checkpoint.get('model_state_dict', checkpoint)
                )
                print("✅ MobileNetV3 model loaded successfully")

            except Exception as e:
                print(f"⚠️ MobileNet load failed ({e}), trying ResNet34 fallback...")
                self.model = models.resnet34(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
                self.model.load_state_dict(
                    checkpoint.get('model_state_dict', checkpoint),
                    strict=False
                )
                print("✅ ResNet34 model loaded with partial weights")

            self.model.eval()

        except Exception as e:
            print(f"❌ Model load error: {e}")
            self.model = None

    def predict_webcam(self):
        if not self.model:
            print("⚠️ No valid model loaded, running in test mode (no predictions).")

        cap = None
        for i in [0, 1, 2]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"📹 Camera detected at index {i}")
                break
            cap.release()

        if not cap or not cap.isOpened():
            print("❌ No webcam found. Check connection.")
            return

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
                    break

                frame_count += 1
                if frame_count % 5 != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(frame_rgb).unsqueeze(0)

                if self.model:
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, predicted = torch.max(probabilities, 0)

                    animal = self.class_names[predicted.item()]
                    conf = confidence.item()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {animal} ({conf:.1%})")

                    if conf > self.confidence_threshold and self.bot_token:
                        now = time.time()
                        if now - self.last_sent_time > 10:
                            self.send_to_telegram(frame, animal, conf)
                            self.last_sent_time = now

                    if conf > 0.7:
                        filename = os.path.join(
                            self.output_dir,
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{animal}_{conf:.2f}.jpg"
                        )
                        cv2.imwrite(filename, frame)

                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running test frame — no model loaded.")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            print("📦 Camera released. Exiting...")

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
            print(f"📱 Sent {animal} alert to Telegram")
        except Exception as e:
            print(f"❌ Telegram error: {e}")


if __name__ == "__main__":
    detector = LiteAnimalDetector()
    detector.predict_webcam()
