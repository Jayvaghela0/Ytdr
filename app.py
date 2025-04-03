import os
import torch
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "weights/swinir_lightweight_x2.pth"
os.makedirs("weights", exist_ok=True)

# Load model from local weights
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model weights not found! Build process failed.")
    
    model = torch.hub.load('JingyunLiang/SwinIR', 'swinir', pretrained=False)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    return model.eval()

# Initialize model at startup
try:
    model = load_model()
    print("Model loaded successfully from local weights")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    model = None

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if model is None:
        return jsonify({"error": "Model not available"}), 503
        
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        
        with torch.no_grad():
            output = model(img_tensor)
        
        result = (output.squeeze().permute(1,2,0).clamp(0,1).numpy() * 255).astype('uint8')
        enhanced_img = Image.fromarray(result)
        
        img_io = BytesIO()
        enhanced_img.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
