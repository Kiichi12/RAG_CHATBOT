from transformers import pipeline
from sentence_transformers import CrossEncoder
import os
import torch
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
from rank_bm25 import BM25Okapi

app = Flask(__name__)
CORS(app)

# Configuration
file_path = "quiz_details.txt"
relevance_threshold = 0.3
bm25_k1 = 1.5
bm25_b = 0.75
top_k = 3  # Keep small for memory efficiency
model_name = "google/flan-t5-base"  # Using base model

# Global variables
lines = []
bm25 = None
reranker = None
llm_pipeline = None

def preprocess_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def load_models():
    global lines, bm25, reranker
    
    # Load document
    lines = preprocess_document(file_path)
    
    # Initialize BM25 with tokenized corpus
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in lines]
    bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)
    
    # Use tiny cross-encoder to save memory
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2", max_length=512)

def initialize_llm():
    return pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        model_kwargs={
            "cache_dir": "model_cache",
            "low_cpu_mem_usage": True
        }
    )

def retrieve_documents(query):
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[-top_k:]
    return [lines[i] for i in top_indices]

# Replace the generate_answer function with:
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def generate_answer(query, context):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={
            "inputs": prompt,
            "parameters": {"max_length": 200}
        }
    )
    return response.json()[0]['generated_text']

# def generate_answer(query, context):
#     prompt = f"""
#     Context: {context}
#     Question: {query}
#     Answer concisely using ONLY the context. If unanswerable say: "Please consult INFINITUM 2025 organizers."
#     """
    
#     return llm_pipeline(
#         prompt,
#         max_length=200,  # Keep short for memory efficiency
#         num_beams=2,     # Reduce from default (4) to save memory
#         num_return_sequences=1,
#         early_stopping=True
#     )[0]['generated_text']

@app.route('/api/init', methods=['POST'])
def initialize():
    global llm_pipeline
    try:
        # Only download what's needed
        nltk.download('punkt', quiet=True)
        
        # Load models sequentially to avoid memory spike
        load_models()
        
        # Initialize LLM last since it's the largest
        llm_pipeline = initialize_llm()
        
        return jsonify({"message": "Models loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sendprompt', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Memory-efficient retrieval
        candidates = retrieve_documents(user_input)
        if not candidates:
            return jsonify({"response": "No relevant information found"}), 200

        # Rerank with tiny model
        pairs = [(user_input, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        best_idx = np.argmax(scores)
        best_doc = candidates[best_idx]
        
        # Skip similarity check to save computation
        if scores[best_idx] < relevance_threshold:
            return jsonify({"response": "No relevant answer found"}), 200

        # Generate with memory constraints
        answer = generate_answer(user_input, best_doc)
        return jsonify({"answer": answer, "context": best_doc}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return 'Service Online'

if __name__ == "__main__":
    # Enable garbage collection aggressively
    import gc
    gc.enable()
    gc.set_threshold(10, 5, 5)  # (threshold0, threshold1, threshold2)
    
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)