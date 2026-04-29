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