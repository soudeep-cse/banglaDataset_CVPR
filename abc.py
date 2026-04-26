from openai import OpenAI

client = OpenAI(
    base_url="https://ha36csd7jtcgyj-11434.proxy.runpod.net/v1",
    api_key="ollama"
)

# Text
response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)



#
#  import requests
# import base64

# # লোকাল Ollama সার্ভার
# OLLAMA_HOST = "http://localhost:11434"

# # টেস্ট ইমেজ পাথ
# image_path = r"C:\Users\s.soudeep\Documents\banglaDataset_CVPR\Bangla-Bayanno-full\images\COCO_train2014_000000000144.jpg"

# try:
#     # ইমেজকে base64 করুন
#     with open(image_path, "rb") as f:
#         image_data = base64.b64encode(f.read()).decode("utf-8")

#     print("📸 ইমেজ লোড হয়েছে। Ollama-তে পাঠাচ্ছি...")
    
#     # qwen2.5vl দিয়ে রিকোয়েস্ট করুন
#     response = requests.post(
#         f"{OLLAMA_HOST}/api/chat",
#         json={
#             "model": "qwen2.5vl:latest",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "What is in this image? Answer in one sentence.",
#                     "images": [image_data]
#                 }
#             ],
#             "stream": False
#         },
#         timeout=120  # বড় ইমেজের জন্য বেশি সময় দিন
#     )

#     result = response.json()
#     print("\n✅ সফল!")
#     print(f"উত্তর: {result.get('message', {}).get('content', 'কোনো উত্তর নেই')}")
    
# except FileNotFoundError:
#     print(f"❌ ইমেজ ফাইল পাওয়া যায়নি: {image_path}")
# except requests.exceptions.ConnectionError:
#     print(f"❌ Ollama সার্ভারে সংযুক্ত হতে পারছি না")
#     print(f"   নিশ্চিত করুন: ollama serve চলছে")
#     print(f"   এবং localhost:11434 এ এক্সেসযোগ্য")
# except Exception as e:
#     print(f"❌ ত্রুটি: {e}")


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