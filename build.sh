#!/bin/bash
# Create weights directory
mkdir -p weights

# Download model weights
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth -O weights/swinir_lightweight_x2.pth

# Install dependencies
pip install -r requirements.txt
