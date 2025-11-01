#!/bin/bash
# Raspberry Pi Installation Script

echo "🍓 Setting up Farm Animal Detection on Raspberry Pi..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv libopencv-dev python3-opencv

# Install USB camera support
sudo apt install -y v4l-utils

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r rasbberypirequirements.txt

# Test camera
echo "📹 Testing USB webcam..."
v4l2-ctl --list-devices

echo "✅ Installation complete!"
echo "🚀 Run: source venv/bin/activate && python run_lite.py"