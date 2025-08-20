import json
import requests
import ollama

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2:3b",
    "prompt": "What is the capital of France?"
}

res = requests.post(url, json=data, stream=False)

if res.status_code == 200:
    print("Generated text:", end=" ", flush=True)
    for line in res.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", res.status_code, res.text)