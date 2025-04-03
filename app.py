import os
import time
import threading
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'temp_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model loading setup
model = None
model_lock = threading.Lock()

def load_model():
    """Thread-safe model loader"""
    global model
    with model_lock:
        if model is None:
            print("Loading SwinIR model...")
            model = torch.hub.load(
                'mateuszbuda/brain-segmentation-pytorch',
                'swin_ir',
                pretrained=True
            ).eval()
            # Quantize model to reduce size
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d}, dtype=torch.qint8
            )
            print("Model loaded successfully")

# Initialize model at startup
load_model()

def clean_old_files():
    """Delete files older than 3 minutes"""
    while True:
        now = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < now - 180:
                os.remove(filepath)
        time.sleep(60)  # Check every minute

# Start cleanup thread
cleaner_thread = threading.Thread(target=clean_old_files, daemon=True)
cleaner_thread.start()

@app.route('/download', methods=['POST'])
def download_video():
    if 'url' not in request.json:
        return jsonify({"error": "No URL provided"}), 400
    
    youtube_url = request.json['url']
    
    try:
        # Process image
        with torch.no_grad():
            # Convert URL to tensor (example - adapt for your actual processing)
            img_tensor = torch.rand((3, 256, 256))  # Replace with actual image processing
            output = model(img_tensor.unsqueeze(0))
            
        # Save to temporary file
        timestamp = str(int(time.time()))
        output_path = os.path.join(UPLOAD_FOLDER, f"output_{timestamp}.jpg")
        torch.save(output, output_path)  # Replace with actual image saving
        
        # Return the processed file
        return send_file(output_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "temp_files": len(os.listdir(UPLOAD_FOLDER))
    })

if __name__ == '__main__':
    # Render-compatible port configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
