import requests
import json
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    
    print("\n" + "="*80)
    print("TESTING TB DETECTION API")
    print("="*80 + "\n")
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint (GET /)...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("   ✓ Root endpoint working!")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"   ✗ Error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "-"*80 + "\n")
    
    # Test 2: Health check
    print("2. Testing health check (GET /health)...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("   ✓ Health check passed!")
            data = response.json()
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
        else:
            print(f"   ✗ Error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "-"*80 + "\n")
    
    # Test 3: Single prediction (if you have a test image)
    print("3. Testing single prediction (POST /predict)...")
    
    # Try to find a test image
    test_image_paths = [
        "test_xray.jpg",
        "test_xray.png",
        "sample.jpg",
        "sample.png"
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if test_image:
        print(f"   Using test image: {test_image}")
        try:
            with open(test_image, "rb") as f:
                files = {"file": (test_image, f, "image/jpeg")}
                response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code == 200:
                print("   ✓ Prediction successful!")
                data = response.json()
                print(f"   Prediction: {data['prediction']}")
                print(f"   Confidence: {data['confidence']}%")
                print(f"   Probabilities: {data['probabilities']}")
            else:
                print(f"   ✗ Error: {response.status_code}")
                print(f"   Message: {response.text}")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    else:
        print("   ⚠ No test image found. Skipping prediction test.")
        print("   To test predictions, place an X-ray image in the current directory")
        print("   and name it 'test_xray.jpg' or 'test_xray.png'")
    
    print("\n" + "-"*80 + "\n")
    
    # Test 4: TTA prediction (if test image exists)
    if test_image:
        print("4. Testing TTA prediction (POST /predict-with-tta)...")
        try:
            with open(test_image, "rb") as f:
                files = {"file": (test_image, f, "image/jpeg")}
                response = requests.post(f"{API_URL}/predict-with-tta", files=files)
            
            if response.status_code == 200:
                print("   ✓ TTA prediction successful!")
                data = response.json()
                print(f"   Prediction: {data['prediction']}")
                print(f"   Confidence: {data['confidence']}%")
                print(f"   Method: {data['method']}")
            else:
                print(f"   ✗ Error: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80 + "\n")
    
    print("📖 To view interactive API documentation, visit:")
    print(f"   {API_URL}/docs")
    print("\n")


if __name__ == "__main__":
    test_api()
