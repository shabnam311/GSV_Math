import json
import base64
import requests
from PIL import Image, ImageDraw
import io

# Point 33: Smoke test for local API or Space API
API_URL = "http://127.0.0.1:7860/predict"

def create_test_image():
    img = Image.new('RGB', (100, 100), color='white')
    d = ImageDraw.Draw(img)
    d.text((10,10), "Triangle", fill="black")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def test_api():
    print("Running smoke test...")
    payload = {
        "image": create_test_image(),
        "question": "What is in the image?",
        "num_samples": 1
    }
    try:
        resp = requests.post(API_URL, json=payload)
        print("Status Code:", resp.status_code)
        print("Response:", json.dumps(resp.json(), indent=2))
    except Exception as e:
        print("Error contacting API:", e)

if __name__ == "__main__":
    test_api()
