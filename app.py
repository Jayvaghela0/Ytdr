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

# Load model architecture from official repo
def load_model():
    model = torch.hub.load('JingyunLiang/SwinIR', 'swinir', pretrained=False)
    state_dict = torch.load(MODEL_WEIGHTS)
    model.load_state_dict(state_dict)
    return model.eval()

# Global model instance with lazy loading
model = None
model_lock = threading.Lock()

@app.before_first_request
def initialize_model():
    global model
    with model_lock:
        if model is None:
            print("Downloading SwinIR weights...")
            if not os.path.exists(MODEL_WEIGHTS):
                torch.hub.download_url_to_file(
                    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
                    MODEL_WEIGHTS
                )
            model = load_model()
            print("Model loaded successfully")

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        
        with torch.no_grad():
            enhanced = model(img_tensor)
        
        enhanced_img = (enhanced.squeeze().permute(1,2,0).clamp(0,1).numpy() * 255).astype('uint8')
        result = Image.fromarray(enhanced_img)
        
        img_io = BytesIO()
        result.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
