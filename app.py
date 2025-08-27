from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load MONAI DenseNet model (same as your working app)
try:
    from monai.networks.nets import DenseNet121
    
    # Create MONAI DenseNet121 model
    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=4,  # 4 classes
        pretrained=False
    )
    
    # Load trained weights
    checkpoint = torch.load(r'D:\Amit Data\Amit data\Foot Deformities Detection\Model\monai_densenet_efficient.pth', 
                           map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ MONAI DenseNet model loaded successfully")
    
except ImportError:
    print("⚠️ MONAI not available. Install with: pip install monai")
    model = None
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    with open('smartfoot.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/docs')
def docs():
    with open('documentation.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        if model:
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Debug: print all probabilities
                print(f"Raw output: {output}")
                print(f"Probabilities: {probabilities}")
                print(f"Predicted class index: {predicted_class}")
            
            # Try different class mappings - the order might be different in your training
            possible_class_orders = [
                ['Flat Foot', 'Hallux Valgus', 'Ulcer', 'Normal'],
                ['Normal', 'Flat Foot', 'Hallux Valgus', 'Ulcer'],
                ['Flat Foot', 'Normal', 'Hallux Valgus', 'Ulcer'],
                ['Ulcer', 'Hallux Valgus', 'Flat Foot', 'Normal']
            ]
            
            # Use correct class mapping from your working app
            classes = ['Normal', 'Flatfoot', 'Foot Ulcer', 'Hallux Valgus']
            result = f"Diagnosis: {classes[predicted_class]} (Confidence: {confidence:.1%})"
        else:
            # Demo mode with filename-based prediction for testing
            filename_lower = filename.lower()
            if 'flatfoot' in filename_lower or 'flat' in filename_lower:
                result = "Diagnosis: Flat Foot (Demo Mode - 85.0%)"
            elif 'hallux' in filename_lower or 'bunion' in filename_lower:
                result = "Diagnosis: Hallux Valgus (Demo Mode - 85.0%)"
            elif 'ulcer' in filename_lower:
                result = "Diagnosis: Ulcer (Demo Mode - 85.0%)"
            else:
                result = "Diagnosis: Normal (Demo Mode - 85.0%)"
        
        return jsonify({'result': result, 'filename': filename})

if __name__ == '__main__':
    app.run(debug=True)