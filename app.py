from flask import Flask, render_template, request
import faiss
import numpy as np

app = Flask(__name__)

# Load FAISS index once at startup
faiss_index = faiss.read_index("your_index_file.index")  # Make sure this file exists

# Dummy embedding function – replace with your actual model logic
def get_embedding(text):
    # Replace this with actual embedding logic (e.g., OpenAI, HuggingFace, etc.)
    return np.random.rand(1, faiss_index.d)  # Random vector for demonstration

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        q_emb = get_embedding(query).astype('float32')  # Ensure it's float32
        D, I = faiss_index.search(q_emb, 2)  # Get top 2 matches
        return f"Top results (Indices): {I[0]}, Distances: {D[0]}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
