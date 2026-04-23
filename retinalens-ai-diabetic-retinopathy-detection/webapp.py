import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import timm

# =========================
# APP
# =========================
app = Flask(__name__, static_folder='.')
CORS(app)

# =========================
# CONFIG (SENİN MODELE GÖRE)
# =========================
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 224   # ✅ SENİN TRAIN
NUM_CLASSES = 5    # ✅ SENİN DATASET

CLASS_NAMES = [
    "Healthy",
    "Mild DR",
    "Moderate DR",
    "Proliferate DR",
    "Severe DR"
]

# =========================
# MODEL LOAD
# =========================
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("✅ Model loaded!")
    return model

model = load_model()

# =========================
# TRANSFORM (TRAIN İLE AYNI)
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# GRAD-CAM
# =========================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.blocks[-1]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1,2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().detach().numpy()

gradcam = GradCAM(model)

# =========================
# ROUTE
# =========================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')

        original = np.array(image)

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item() * 100

        # gradcam
        cam = gradcam.generate(input_tensor, pred)
        cam = cv2.resize(cam, (224,224))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        overlay = heatmap * 0.4 + cv2.resize(original, (224,224))

        _, buffer = cv2.imencode('.png', overlay)
        heatmap_base64 = base64.b64encode(buffer).decode()

        return jsonify({
            "prediction": CLASS_NAMES[pred],
            "confidence": round(confidence, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(probs[i].item()*100,2)
                for i in range(NUM_CLASSES)
            },
            "heatmap": heatmap_base64
        })

    except Exception as e:
        return jsonify({
    "result": CLASS_NAMES[pred],
    "confidence": round(confidence, 2),
    "probabilities": {
        CLASS_NAMES[i]: round(probs[i].item()*100, 2)
        for i in range(len(CLASS_NAMES))
    },
    "heatmap": heatmap_base64
})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 Server running at http://localhost:5000")
    app.run(debug=True)