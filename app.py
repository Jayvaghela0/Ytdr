import os
import time
from threading import Thread
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import torch
import torch.hub
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'temp_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model at startup (quantized for performance)
@app.before_first_request
def load_model():
    app.model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'swin_ir',
        pretrained=True
    ).eval()
    # Quantize model to reduce size without quality loss
    app.model = torch.quantization.quantize_dynamic(
        app.model, {torch.nn.Conv2d}, dtype=torch.qint8
    )

def clean_old_files():
    """Delete files older than 3 minutes"""
    while True:
        now = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.getmtime(filepath) < now - 180:  # 3 minutes
                os.remove(filepath)
        time.sleep(60)  # Check every minute

# Start cleaner thread
Thread(target=clean_old_files, daemon=True).start()

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Process image
        img_file = request.files['image']
        img = Image.open(img_file.stream).convert('RGB')
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        
        # Enhance with SwinIR (full quality)
        with torch.no_grad():
            output = app.model(img_tensor.to('cpu'))
        
        # Convert back to image
        result = (output.squeeze().permute(1,2,0).clamp(0,1).numpy() * 255).astype('uint8')
        enhanced_img = Image.fromarray(result)
        
        # Save to temporary file
        timestamp = str(int(time.time()))
        output_path = os.path.join(UPLOAD_FOLDER, f"enhanced_{timestamp}.jpg")
        enhanced_img.save(output_path, 'JPEG', quality=100)  # 100% quality
        
        # Return the enhanced image
        return send_file(output_path, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
