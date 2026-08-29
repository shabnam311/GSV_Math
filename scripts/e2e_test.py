import json
import base64
import requests
from PIL import Image, ImageDraw
import io

API_URL = "https://shabnam311--gsv-math-backend-gsvmathmodel-solve.modal.run"

def create_test_image():
    img = Image.new('RGB', (100, 100), color='white')
    d = ImageDraw.Draw(img)
    d.text((10,10), "Triangle", fill="black")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def test_api():
    print("Running smoke test against Modal endpoint...")
    payload = {
        "image_base64": create_test_image(),
        "question": "What is in the image?",
        "num_samples": 1
    }
    try:
        resp = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        print("Status Code:", resp.status_code)
        print("Response:", json.dumps(resp.json(), indent=2))
    except Exception as e:
        print("Error contacting API:", e)

if __name__ == "__main__":
    test_api()
