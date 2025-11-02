import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import requests
import time
from datetime import datetime
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
            # Default class names from dataset
            self.class_names = ['Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer', 
                               'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit', 
                               'Skunk', 'Tiger', 'Wild Boar']
            print("⚠️ Using default class names")
        
        print(f"📋 Classes: {self.class_names}")
        
        if 'accuracy' in checkpoint:
            print(f"🎯 Model accuracy: {checkpoint['accuracy']:.4f}")
        else:
            print("📊 Model accuracy: Not available")
        
        # Create model matching the saved architecture
        self.model = models.resnet50(weights=None)
        
        # Try to determine the correct architecture from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check if it's the simple ResNet34 architecture
        if 'fc.weight' in state_dict and len(state_dict['fc.weight'].shape) == 2:
            # Simple linear layer (ResNet34 from final_trainer.py)
            self.model = models.resnet34(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        else:
            # Complex architecture - try to match the saved one
            fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]
            if 'fc.4.weight' in state_dict:
                # Original saved model architecture
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, len(self.class_names))
                )
            else:
                # Default to simple linear
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        
        # Load model weights (handle different formats)
        try:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Trying ResNet34 architecture...")
            # Fallback to ResNet34
            self.model = models.resnet34(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Model loaded with ResNet34 architecture")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform (matching validation transform from training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path):
        """Predict animal class for a single image"""
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        return self.class_names[predicted.item()], confidence.item()
    
    def get_top_predictions(self, image_path, top_k=3):
        """Get top-k predictions for an image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            animal = self.class_names[top_indices[i].item()]
            confidence = top_probs[i].item()
            results.append((animal, confidence))
        
        return results
    
    def send_to_telegram(self, frame, animal, confidence):
        """Send detection image to Telegram"""
        try:
            # Save frame temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{animal}_{confidence:.3f}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Send photo to Telegram
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            with open(filename, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': f"🐾 Animal Detected!\n🔍 Species: {animal}\n📊 Confidence: {confidence:.1%}\n🕰 Time: {datetime.now().strftime('%H:%M:%S')}"
                }
                
                response = requests.post(url, files=files, data=data, timeout=10)
                
                if response.status_code == 200:
                    print(f"📤 Sent {animal} detection to Telegram")
                else:
                    print(f"❌ Telegram error: {response.status_code}")
            
            # Clean up temp file
            import os
            os.remove(filename)
            
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
    
    def predict_webcam(self):
        # Try different camera indices and backends
        cap = None
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        
        for backend in backends:
            for i in range(3):
                print(f"Trying camera {i} with backend {backend}...")
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Wait for camera to initialize
                    import time
                    time.sleep(2)
                    
                    # Try multiple frames
                    for attempt in range(10):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.max() > 0:
                            print(f"✅ Using camera {i} with backend {backend}")
                            break
                        time.sleep(0.1)
                    
                    if ret and test_frame is not None and test_frame.max() > 0:
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap:
                        cap.release()
                        cap = None
            
            if cap and cap.isOpened():
                break
        
        if not cap or not cap.isOpened():
            print("❌ Error: Could not open any webcam")
            print("💡 Fixes to try:")
            print("   1. sudo usermod -a -G video $USER && logout/login")
            print("   2. sudo modprobe uvcvideo")
            print("   3. Check if camera is used by another app")
            return
        
        print("📹 Webcam started. Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Predict on frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            animal = self.class_names[predicted.item()]
            conf = confidence.item()
            
            # Color coding based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Track detection persistence
            current_time = time.time()
            if conf > self.confidence_threshold:
                if self.current_animal == animal:
                    # Same animal detected, check duration
                    if self.detection_start_time and (current_time - self.detection_start_time) >= self.detection_duration:
                        # Animal detected for 5+ seconds, send message
                        if self.bot_token and self.chat_id and (current_time - self.last_sent_time) > self.min_interval:
                            self.send_to_telegram(frame, animal, conf)
                            self.last_sent_time = current_time
                            self.detection_start_time = None  # Reset to avoid spam
                else:
                    # New animal detected, start timer
                    self.current_animal = animal
                    self.detection_start_time = current_time
            else:
                # Low confidence, reset detection
                self.detection_start_time = None
                self.current_animal = None
            
            # Display result with better formatting
            cv2.putText(frame, f'Animal: {animal}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'Confidence: {conf:.3f}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show detection status
            if conf > self.confidence_threshold:
                if self.detection_start_time:
                    elapsed = current_time - self.detection_start_time
                    remaining = max(0, self.detection_duration - elapsed)
                    status = f"Detecting: {remaining:.1f}s left" if remaining > 0 else "📤 Ready to send"
                else:
                    status = "⚠️ Configure Telegram" if not self.bot_token else "Waiting..."
                cv2.putText(frame, status, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save | 5s detection required", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Animal Classification', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'detection_{animal}_{conf:.3f}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Webcam closed")

def test_image_prediction(detector, image_path):
    """Test prediction on a single image"""
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
        
        # Uncomment to test with specific image
        # test_image_prediction(detector, 'path/to/your/image.jpg')
        
        # Real-time webcam detection
        print("\n🎥 Starting webcam detection...")
        if detector.bot_token and detector.chat_id:
            print(f"🤖 Telegram bot enabled - will send photos when confidence > {detector.confidence_threshold:.0%}")
        else:
            print("⚠️ Telegram not configured - update telegram_config.py")
        
        detector.predict_webcam()
        
    except FileNotFoundError:
        print("❌ Model file 'best_animal_model.pth' not found!")
        print("🔧 Please run 'python high_accuracy_trainer.py' first to train the model.")
    except Exception as e:
        print(f"❌ Error: {e}")