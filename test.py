import requests

url = "http://69.30.85.131:22054/api/chat"

payload = {
    "model": "qwen2.5vl:7b",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json()["message"]["content"])