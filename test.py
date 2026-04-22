import requests

BASE = "http://194.68.245.42:22166"

models = requests.get(f"{BASE}/api/tags", timeout=5).json().get("models", [])
print("Models:", [m["name"] for m in models])

response = requests.post(f"{BASE}/api/generate", json={
    "model": models[0]["name"],
    "prompt": "Say hello in one sentence.",
    "stream": False
}, timeout=30).json().get("response")
print("Response:", response)