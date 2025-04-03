import os
import time
import threading
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_WEIGHTS = "weights/swinir_lightweight_x2.pth"
os.makedirs('weights', exist_ok=True)

# Model loading setup
model = None
model_loaded = False
model_lock = threading.Lock()

def download_model_weights():
    """Download model weights if not present"""
    if not os.path.exists(MODEL_WEIGHTS):
        print("Downloading model weights...")
        torch.hub.download_url_to_file(
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
            MODEL_WEIGHTS
        )

def load_model():
    """Load model with thread safety"""
    global model, model_loaded
    with model_lock:
        if not model_loaded:
            download_model_weights()
            print("Loading SwinIR model...")
            model = torch.hub.load('JingyunLiang/SwinIR', 'swinir', pretrained=False)
            state_dict = torch.load(MODEL_WEIGHTS)
            model.load_state_dict(state_dict)
            model = model.eval()
            # Quantize for better performance
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d}, dtype=torch.qint8
            )
            model_loaded = True
            print("Model loaded successfully")

# Initialize model at app startup
load_model()

def clean_old_files():
    """Delete files older than 3 minutes"""
    while True:
        now = time.time()
        for filename in os.listdir('temp_images'):
            filepath = os.path.join('temp_images', filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < now - 180:
                os.remove(filepath)
        time.sleep(60)  # Check every minute

# Start cleanup thread
os.makedirs('temp_images', exist_ok=True)
cleaner_thread = threading.Thread(target=clean_old_files, daemon=True)
cleaner_thread.start()

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """Image enhancement endpoint"""
    if not model_loaded:
        return jsonify({"error": "Model not loaded yet"}), 503
        
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Process image
        img = Image.open(request.files['image'].stream).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        
        # Enhance image
        with torch.no_grad():
            enhanced = model(img_tensor)
        
        # Prepare output
        enhanced_img = (enhanced.squeeze().permute(1,2,0).clamp(0,1).numpy() * 255).astype('uint8')
        result = Image.fromarray(enhanced_img)
        
        # Save temporarily (auto-cleaned later)
        timestamp = str(int(time.time()))
        output_path = os.path.join('temp_images', f"enhanced_{timestamp}.jpg")
        result.save(output_path, 'JPEG', quality=95)
        
        return send_file(output_path, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Service status endpoint"""
    return jsonify({
        "status": "running",
        "model_loaded": model_loaded,
        "temp_files": len(os.listdir('temp_images'))
    })

if __name__ == '__main__':
    # Render-compatible configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
