from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://69.30.85.131:22054/v1",
    api_key="ollama"
)

with open(r"C:\Users\slmns\Desktop\thesis\banglaDataset_CVPR\data\images\COCO_train2014_000000000144.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="qwen2.5vl:7b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": "What do you see in this image?"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)