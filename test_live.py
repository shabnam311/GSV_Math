import requests

url = "https://shabnam-offl--gsv-math-backend-gsvmathmodel-solve.modal.run"

# Test 1: OPTIONS request for CORS
headers = {
    "Origin": "https://gsv-math.vercel.app",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "X-API-Key, Content-Type"
}
r1 = requests.options(url, headers=headers)
print("OPTIONS Status:", r1.status_code)
print("OPTIONS Headers:", r1.headers)

# Test 2: POST request
r2 = requests.post(url, json={"question": "test"}, headers={"X-API-Key": "dev-secret-key"})
print("POST Status:", r2.status_code)
print("POST Response:", r2.text)
