# 🐾 Farm Animal Detection System

Real-time animal classification system with Telegram alerts for farm safety monitoring.

## 📋 Project Summary

**Core Features:**
- 🎯 **15 Animal Classes:** Armadilles, Bear, Birds, Cow, Crocodile, Deer, Elephant, Goat, Horse, Jaguar, Monkey, Rabbit, Skunk, Tiger, Wild Boar
- 🚀 **GPU Accelerated:** CUDA support for fast training and inference
- 📱 **Telegram Integration:** Auto-sends photos when confidence > 90%
- 🎥 **Real-time Detection:** Live webcam monitoring
- 🧠 **High Accuracy:** ResNet50 with custom classifier achieving 95%+ accuracy

## 📁 Project Structure

```
farm_data/
├── train/          # Training images (2,287 total)
└── val/            # Validation images (559 total)

high_accuracy_trainer.py    # Main training script (50 epochs)
run.py                     # Real-time detection with Telegram
telegram_config.py         # Bot configuration
best_animal_model.pth      # Trained model weights
requirements.txt           # Full dependencies
rasbberypirequirements.txt # Raspberry Pi optimized
```

## 🚀 Quick Start

### 1. Training (Desktop/GPU)
```bash
pip install -r requirements.txt
python high_accuracy_trainer.py
```

### 2. Detection (Desktop/Raspberry Pi)
```bash
# Desktop
pip install -r requirements.txt

# Raspberry Pi
pip install -r rasbberypirequirements.txt

python run.py
```

### 3. Telegram Setup
Update `telegram_config.py`:
```python
BOT_TOKEN = "your_bot_token"
CHAT_ID = "your_chat_id"
CONFIDENCE_THRESHOLD = 0.9  # 90%
```

## 🎯 Model Performance

- **Architecture:** ResNet50 + Custom Classifier
- **Training:** 50 epochs with data augmentation
- **Accuracy:** 95%+ validation accuracy
- **Classes:** 15 farm animals with class balancing
- **Inference:** Real-time on GPU/CPU

## 📱 Telegram Features

- 🚨 **Auto Alerts:** Photos sent when confidence > 90%
- ⏱️ **Rate Limited:** 5-second intervals between messages
- 📊 **Rich Info:** Animal name, confidence, timestamp
- 🎨 **Color Coded:** Green/Yellow/Red confidence display

## 🔧 Hardware Requirements

**Training:**
- NVIDIA GPU (RTX 4050+)
- 8GB+ RAM
- Ubuntu/Linux

**Inference:**
- Raspberry Pi 4+ or Desktop
- USB Camera/Webcam
- 4GB+ RAM

## 📦 Key Files

- `high_accuracy_trainer.py` - Complete training pipeline
- `run.py` - Real-time detection with Telegram
- `telegram_config.py` - Bot configuration
- `best_animal_model.pth` - Trained model (98MB)

## 🎮 Controls

**Webcam Mode:**
- `q` - Quit
- `s` - Save screenshot
- Auto Telegram alerts at 90%+ confidence

## 🔒 Security Features

- Class imbalance handling
- Confidence thresholding
- Rate limiting for alerts
- Error handling and recovery