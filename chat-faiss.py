import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import tiktoken  # For token counting (OpenAI tokenizer compatible)

# Mistral API key
MISTRAL_API_KEY = "cRXfDbOI8iauSUlzCY8PWHZoUA5x8X0u"

# Load business data JSON
def load_business_data(file_path="business_data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Flatten JSON data into list of text chunks
def json_to_chunks(data):
    chunks = []
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value)
        chunks.append(f"{key}: {value}")
    return chunks

# Initialize sentence transformer model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS index
def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Encode prompt tokens count using tiktoken (approximation)
def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(text)
        return len(tokens)
    except Exception:
        # fallback simple split
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
    result = response.json()
    return result

def main():
    data = load_business_data()
    chunks = json_to_chunks(data)
    index, embeddings = build_faiss_index(chunks)

    print("Business data loaded and vectorized. Ready to answer questions!")

    while True:
        question = input("\nApna sawaal likho (exit ke liye 'exit'): ")
        if question.lower() == "exit":
            break

        # Embed the question
        q_emb = embed_model.encode([question], convert_to_numpy=True)

        # Search in index
        k = 2  # top 2 relevant chunks
        D, I = index.search(q_emb, k)

        relevant_chunks = [chunks[i] for i in I[0]]

        # Build prompt for Mistral
        context_text = "\n".join(relevant_chunks)
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer based on the above context."

        # Count tokens sent to Mistral
        tokens_sent = count_tokens(prompt)

        # Call Mistral
        response = call_mistral_api(prompt)
        answer = response["choices"][0]["message"]["content"]

        # Count tokens received from Mistral
        tokens_received = count_tokens(answer)

        # Print details
        print("\n--- Mistral Response ---")
        print(f"Tokens sent: {tokens_sent}")
        print(f"Tokens received: {tokens_received}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

