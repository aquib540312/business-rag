
from flask import Flask, request, render_template
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import tiktoken
import os

app = Flask(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_business_data(file_path="business_data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_to_chunks(data):
    chunks = []
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value)
        chunks.append(f"{key}: {value}")
    return chunks

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks, embeddings

def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(text)
        return len(tokens)
    except Exception:
        return len(text.split())

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
    return response.json()

data = load_business_data()
chunks = json_to_chunks(data)
index, all_chunks, embeddings = build_faiss_index(chunks)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    tokens_sent = 0
    tokens_received = 0
    if request.method == "POST":
        question = request.form["question"]
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        D, I = index.search(q_emb, 2)
        relevant_chunks = [all_chunks[i] for i in I[0]]
        context_text = "\n".join(relevant_chunks)
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer based on the above context."
        tokens_sent = count_tokens(prompt)
        response = call_mistral_api(prompt)
        answer = response["choices"][0]["message"]["content"]
        tokens_received = count_tokens(answer)
    return render_template("index.html", answer=answer, tokens_sent=tokens_sent, tokens_received=tokens_received)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
