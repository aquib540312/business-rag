from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
import tiktoken

# Flask app
app = Flask(__name__)

# Mistral API Key
MISTRAL_API_KEY = "cRXfDbOI8iauSUlzCY8PWHZoUA5x8X0u"

# Load business data
with open("business_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to chunks
def json_to_chunks(data):
    chunks = []
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value)
        chunks.append(f"{key}: {value}")
    return chunks

chunks = json_to_chunks(data)

# Embedding model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Token counter
def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except:
        return len(text.split())

# Call Mistral API
def call_mistral_api(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Web interface
@app.route("/")
def home():       # 👈 Just rename this function
    return render_template("index.html")

# API endpoint
@app.route("/api", methods=["POST"])
def get_answer():
    question = request.json.get("question", "")
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k=2)
    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer based on the above context."

    answer = call_mistral_api(prompt)
    tokens_sent = count_tokens(prompt)
    tokens_received = count_tokens(answer)

    return jsonify({
        "answer": answer,
        "tokens_sent": tokens_sent,
        "tokens_received": tokens_received,
        "context": context
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
