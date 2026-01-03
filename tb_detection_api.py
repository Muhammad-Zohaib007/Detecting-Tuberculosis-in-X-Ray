"""
FastAPI Application for TB Detection - ULTRA SIMPLE TTA VERSION
This version uses the most basic approach to avoid any transform issues
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
import io
import numpy as np
from typing import Dict
import uvicorn

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, pretrained=False):
        super(EfficientNetB3Model, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# SETUP
# ============================================================================

app = FastAPI(title="TB Detection API", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = ["Healthy", "TB"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.on_event("startup")
async def load_model():
    global model
    try:
        print("Loading model...")
        model = EfficientNetB3Model(num_classes=2, dropout=0.5, pretrained=False)
        model.load_state_dict(torch.load("tb_detection_efficientnet_b3.pth", map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(pil_image):
    """Convert PIL image to tensor"""
    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor.to(device)


def predict_single(img_tensor):
    """Get prediction from single tensor"""
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
    return probs.cpu().numpy()[0]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "TB Detection API",
        "version": "1.0.2 (Simple TTA)",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """Single prediction without TTA"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess and predict
        img_tensor = preprocess_image(image)
        probs = predict_single(img_tensor)
        
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        
        return {
            "success": True,
            "prediction": class_names[predicted],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "healthy": round(float(probs[0]) * 100, 2),
                "tb": round(float(probs[1]) * 100, 2)
            }
        }
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-with-tta")
async def predict_with_tta(file: UploadFile = File(...)) -> Dict:
    """Prediction with Test-Time Augmentation - SIMPLE VERSION"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        print("Reading image...")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"Image loaded: {image.size}")
        
        # Store all predictions
        all_probs = []
        
        # 1. Original image
        print("Processing original...")
        img_tensor = preprocess_image(image)
        probs = predict_single(img_tensor)
        all_probs.append(probs)
        print(f"Original probs: {probs}")
        
        # 2. Horizontal flip
        print("Processing flipped...")
        image_flipped = ImageOps.mirror(image)
        img_tensor = preprocess_image(image_flipped)
        probs = predict_single(img_tensor)
        all_probs.append(probs)
        print(f"Flipped probs: {probs}")
        
        # 3. Rotated +5 degrees
        print("Processing rotated +5...")
        image_rotated = image.rotate(5, expand=False, fillcolor=(0, 0, 0))
        img_tensor = preprocess_image(image_rotated)
        probs = predict_single(img_tensor)
        all_probs.append(probs)
        print(f"Rotated +5 probs: {probs}")
        
        # 4. Rotated -5 degrees
        print("Processing rotated -5...")
        image_rotated = image.rotate(-5, expand=False, fillcolor=(0, 0, 0))
        img_tensor = preprocess_image(image_rotated)
        probs = predict_single(img_tensor)
        all_probs.append(probs)
        print(f"Rotated -5 probs: {probs}")
        
        # 5. Brightness adjusted
        print("Processing brightness...")
        # Simple brightness adjustment
        image_array = np.array(image).astype(np.float32)
        image_array = np.clip(image_array * 1.1, 0, 255).astype(np.uint8)
        image_bright = Image.fromarray(image_array)
        img_tensor = preprocess_image(image_bright)
        probs = predict_single(img_tensor)
        all_probs.append(probs)
        print(f"Bright probs: {probs}")
        
        # Average all predictions
        print("Averaging predictions...")
        avg_probs = np.mean(all_probs, axis=0)
        predicted = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted])
        
        print(f"Final averaged probs: {avg_probs}")
        print(f"Prediction: {class_names[predicted]} ({confidence*100:.2f}%)")
        
        return {
            "success": True,
            "prediction": class_names[predicted],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "healthy": round(float(avg_probs[0]) * 100, 2),
                "tb": round(float(avg_probs[1]) * 100, 2)
            },
            "method": "Test-Time Augmentation (5 augmentations)",
            "augmentations": ["Original", "Horizontal Flip", "Rotate +5°", "Rotate -5°", "Brightness +10%"],
            "details": {
                "image_size": list(image.size),
                "device": str(device)
            }
        }
    
    except Exception as e:
        print(f"ERROR in TTA: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> Dict:
    """Batch prediction"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images")
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = preprocess_image(image)
            probs = predict_single(img_tensor)
            
            predicted = int(np.argmax(probs))
            
            results.append({
                "filename": file.filename,
                "prediction": class_names[predicted],
                "confidence": round(float(probs[predicted]) * 100, 2),
                "probabilities": {
                    "healthy": round(float(probs[0]) * 100, 2),
                    "tb": round(float(probs[1]) * 100, 2)
                }
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TB DETECTION API - Ultra Simple TTA Version")
    print("="*80)
    print(f"Device: {device}")
    print("\nStarting server on http://localhost:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
