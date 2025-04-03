#!/bin/bash
# Create required directories
mkdir -p weights
mkdir -p temp_images

# Download model weights from reliable mirror
wget https://huggingface.co/spaces/akhaliq/SwinIR/resolve/main/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
    -O weights/swinir_lightweight_x2.pth

# Verify download succeeded
if [ ! -f "weights/swinir_lightweight_x2.pth" ]; then
    echo "Error: Model weights download failed!"
    exit 1
fi

# Install dependencies
pip install -r requirements.txt
