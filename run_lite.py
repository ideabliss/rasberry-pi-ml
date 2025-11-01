import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import requests
import time
from datetime import datetime

# Lightweight version for 2GB RAM
class LiteAnimalDetector:
    def __init__(self, model_path='best_animal_model.pth'):
        # Force CPU and optimize memory
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        
        self.class_names = ['Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer', 
                           'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit', 
                           'Skunk', 'Tiger', 'Wild Boar']
        
        # Load lightweight model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Use MobileNet for lower memory
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(self.class_names))
            
            # If checkpoint doesn't match, use ResNet34 (smaller than ResNet50)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            except:
                self.model = models.resnet34(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            
            self.model.eval()
            print("✅ Lite model loaded")
            
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return
        
        # Telegram config
        try:
            from telegram_config import BOT_TOKEN, CHAT_ID, CONFIDENCE_THRESHOLD
            self.bot_token = BOT_TOKEN
            self.chat_id = CHAT_ID
            self.confidence_threshold = CONFIDENCE_THRESHOLD
        except:
            self.bot_token = None
            self.chat_id = None
            self.confidence_threshold = 0.9
        
        self.last_sent_time = 0
        
        # Lightweight transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_webcam(self):
        # Try different camera indices for USB webcam
        cap = None
        for camera_id in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"📹 Found camera at index {camera_id}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("❌ No USB webcam found. Check connection.")
            return
        
        # Raspberry Pi optimized settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Very low resolution for Pi
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 10)            # Lower FPS for stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer lag
        
        # Test frame capture
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Cannot capture from webcam")
            cap.release()
            return
        
        print(f"✅ Webcam ready: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        print("📹 Lite detection started (Press 'q' to quit)")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to save memory on Pi
            frame_count += 1
            if frame_count % 5 != 0:
                # Skip processing but still show frame
                cv2.putText(frame, "Processing...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Animal Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Predict
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            animal = self.class_names[predicted.item()]
            conf = confidence.item()
            
            # Color coding
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
            
            # Send to Telegram if high confidence
            if conf > self.confidence_threshold and self.bot_token:
                current_time = time.time()
                if current_time - self.last_sent_time > 10:  # 10 second interval
                    self.send_to_telegram(frame, animal, conf)
                    self.last_sent_time = current_time
            
            # Display with Pi-friendly text
            cv2.putText(frame, f'{animal}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f'{conf:.1%}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show telegram status
            if conf > self.confidence_threshold:
                cv2.putText(frame, "ALERT SENT!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.imshow('Animal Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def send_to_telegram(self, frame, animal, confidence):
        try:
            filename = f"detection_{animal}_{confidence:.2f}.jpg"
            cv2.imwrite(filename, frame)
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            with open(filename, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': f"🐾 {animal} detected!\nConfidence: {confidence:.1%}"
                }
                requests.post(url, files=files, data=data, timeout=5)
            
            import os
            os.remove(filename)
            print(f"📱 Sent {animal} to Telegram")
        except Exception as e:
            print(f"❌ Telegram error: {e}")

if __name__ == "__main__":
    detector = LiteAnimalDetector()
    detector.predict_webcam()