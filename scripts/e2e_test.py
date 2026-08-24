import json
import base64
import requests
from PIL import Image, ImageDraw
import io

# E2E test hitting the live production URL (Point 82)
# TODO: Replace with the actual deployed Space URL
SPACE_URL = "https://username-gsv-math-demo.hf.space/predict"

def create_test_image():
    img = Image.new('RGB', (100, 100), color='white')
    d = ImageDraw.Draw(img)
    d.text((10,10), "x + 2 = 5", fill="black")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def test_live_api():
    print(f"Testing live API at {SPACE_URL}...")
    payload = {
        "image": create_test_image(),
        "question": "Solve for x.",
        "num_samples": 1
    }
    try:
        resp = requests.post(SPACE_URL, json=payload, timeout=90)
        if resp.status_code == 200:
            print("✅ Success! Response:")
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"❌ Failed with status code {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    test_live_api()
