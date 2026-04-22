import requests
import base64

# Image কে base64 করো
with open(r"C:\Users\slmns\Desktop\thesis\banglaDataset_CVPR\BanglaVerse\data\images\culture\images\culture_001.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://194.68.245.42:22166/api/chat",
    json={
        "model": "qwen3.5:9b",
        "messages": [
            {
                "role": "user",
                "content": "What is in this image? no details, just a short answer. no explanation. no need to reason. just answer the question. no need to explain your answer. ",
                "images": [image_data]  # base64 string
            }
        ],
        "stream": False
    }
)

print(response.json())


# import requests
# import base64

# with open(r"C:\Users\slmns\Desktop\thesis\banglaDataset_CVPR\data\images\COCO_train2014_000000000144.jpg", "rb") as f:
#     image_data = base64.b64encode(f.read()).decode("utf-8")

# response = requests.post(
#     "http://194.68.245.42:22166/api/chat",
#     json={
#         "model": "qwen3.5:9b",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is in this image?",
#                 "images": [image_data]
#             }
#         ],
#         "stream": False
#     }
# )

# print(response.json())