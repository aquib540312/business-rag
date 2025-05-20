from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = Flask(__name__)

# Load data
with open("business_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to chunks
def json_to_chunks(data):
    chunks = []
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value)
        chunks.append(f"{key}: {value}")
    return chunks

chunks = json_to_chunks(data)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# FAISS index
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)

@app.route("/")
def home():  # changed name from `index` to `home`
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def get_answer():
    question = request.json.get("question", "")
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = faiss_index.search(q_emb, k=2)
    relevant_chunks = [chunks[i] for i in I[0]]

    # simple answer without API
    context = "\n".join(relevant_chunks)
    fake_answer = f"Based on the info:\n\n{context}\n\nYour Question:\n{question}"

    return jsonify({"answer": fake_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
