# 🍓 Raspberry Pi Setup Guide

Complete setup instructions for Farm Animal Detection System on Raspberry Pi.

## 📋 Prerequisites

- Raspberry Pi 4+ (4GB RAM recommended)
- MicroSD card (32GB+)
- USB Camera or Pi Camera Module
- Internet connection

## 🔧 System Setup

### 1. Initial Configuration
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git cmake libopenblas-dev libatlas-base-dev

# Enable camera
sudo raspi-config
# Interface Options > Camera > Enable > Reboot
```

### 2. Clone Repository
```bash
git clone https://github.com/ideabliss/rasberry-pi-ml.git
cd rasberry-pi-ml
```

### 3. Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Pi-optimized packages
pip install -r rasbberypirequirements.txt
```

### 4. Camera Test
```bash
# Test camera connection
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera OK' if cap.isOpened() else '❌ Camera Error'); cap.release()"
```

## 🚀 Running the System

### Manual Start
```bash
source venv/bin/activate
python3 run_lite.py  # Optimized for Raspberry Pi
```

### Auto-Start Service
```bash
# Create service file
sudo nano /etc/systemd/system/animal-detection.service
```

Service content:
```ini
[Unit]
Description=Farm Animal Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rasberry-pi-ml
Environment=PATH=/home/pi/rasberry-pi-ml/venv/bin
ExecStart=/home/pi/rasberry-pi-ml/venv/bin/python run_lite.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable animal-detection.service
sudo systemctl start animal-detection.service
sudo systemctl status animal-detection.service
```

## 📱 Telegram Configuration

Bot is pre-configured in `telegram_config.py`:
- Token: `7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg`
- Chat ID: `6070190518`
- Threshold: 90%

Test bot:
```bash
curl "https://api.telegram.org/bot7952163683:AAHaY8pogTsIqCfbqpAWSYhsVZ6G_raAFeg/getMe"
```

## 🎮 Controls

- `q` - Quit application
- `s` - Save screenshot
- Auto Telegram alerts at 90%+ confidence

## 🔧 Troubleshooting

### Camera Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Reboot required
sudo reboot
```

### Performance Optimization
```bash
# Increase GPU memory
sudo raspi-config
# Advanced Options > Memory Split > 128

# Monitor resources
htop
vcgencmd measure_temp
```

### Service Management
```bash
# Check logs
sudo journalctl -u animal-detection.service -f

# Restart service
sudo systemctl restart animal-detection.service

# Stop service
sudo systemctl stop animal-detection.service
```

## 📊 Expected Performance

- **Detection Speed:** 2-5 FPS on Pi 4
- **Animals Detected:** 15 classes
- **Accuracy:** 95%+ confidence
- **Telegram Alerts:** Auto-sent at 90%+ confidence

## 🔒 Security Notes

- Bot token is exposed in config file
- Consider environment variables for production
- Rate limiting prevents spam (5-second intervals)

## 📁 File Structure

```
rasberry-pi-ml/
├── run.py                     # Desktop detection script
├── run_lite.py                # Pi-optimized detection (USE THIS)
├── telegram_config.py         # Bot configuration
├── best_animal_model.pth      # Trained model (98MB)
├── rasbberypirequirements.txt # Pi dependencies
└── setup.md                   # This file
```

## ✅ Quick Verification

1. Camera working: `python3 -c "import cv2; print(cv2.__version__)"`
2. Model exists: `ls -la best_animal_model.pth`
3. Service running: `sudo systemctl status animal-detection.service`
4. Bot responding: Test Telegram bot connection

System ready for 24/7 farm monitoring! 🐄🐷🐎