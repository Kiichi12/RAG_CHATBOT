from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pydantic import BaseModel
import os
import torch
# from torch import init_empty_weights, load_checkpoint_and_dispatch
import faiss

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

from rank_bm25 import BM25Okapi
import numpy as np 
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)

lines = []
index = None
model = None
bm25 = None
tokenized_corpus = None
reranker = None
llm_pipeline = None
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

file_path = "quiz_details.txt"
# file_path = "quiz2.txt"
relevance_threshold = 0.3
bm25_k1 = 1.5
bm25_b = 0.75
top_k = 5

model_name = "google/flan-t5-base"

def preprocess_document(file_path):
    if not os.path.exists(file_path):
        print("file_path does not exist")
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        with open(file_path,'r', encoding = 'utf-8') as file:
            return file.readlines()

def load_model_and_index():
    global lines, model, index, bm25, tokenized_corpus, reranker

    lines = preprocess_document(file_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(lines, show_progress_bar = True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    tokenized_corpus = [word_tokenize(doc.lower()) for doc in lines]

    bm25 = BM25Okapi(tokenized_corpus, k1 = bm25_k1, b = bm25_b)

def initialize_llm():
    global model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # quant_config = BitsAndBytesConfig(
    #     load_in_8bit = True,
    #     bnb_4bit_use_double_quant = True,
    #     bnb_4bit_quant_type = "nf4",
    #     bnb_4bit_compute_dtype = torch.float16
    # )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype = "auto",  # Use "auto" to automatically select dtype based on device
        # quantization_config = quant_config,
        # device_map = "auto" if torch.cuda.is_available() else None,
        # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # low_cpu_mem_usage = True
    )

    return pipeline(
        "text2text-generation",
        model = model,
        device = 0 if torch.cuda.is_available() else -1,
        tokenizer = tokenizer
    )

def hybrid_retrieve(query, top_k = top_k):
    global bm25, tokenized_corpus, model, index
    
    tokenized_query = word_tokenize(query.lower())

    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(bm25_scores)[-top_k:]
    bm25_docs = [lines[i] for i in bm25_top_idx]

    query_emb = model.encode([query])
    _, emb_top_idx = index.search(query_emb, top_k)
    emb_docs = [lines[i] for i in emb_top_idx[0]]

    candidates = list(set(bm25_docs + emb_docs))
    return candidates

def rerank(query, candidates):
    global reranker
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    best_idx = int(np.argmax(scores))
    return candidates[best_idx]


def generate_answer(query, context, llm_pipeline, max_length = 500, num_beams = 3):
    # pairs = [(query, ctx) for ctx in contexts]
    # scores = reranker.predict(pairs)
    # best_idx = int(np.argmax(scores))
    # best_context = contexts[best_idx]

    # if not any(word.lower() in best_context.lower() for word in query.split()):
    #     return "Sorry, I cannot answer that based on the current policy"

    # prompt = f"""
    # Context: {best_context}

    # Question: {query}

    # Answer ONLY using the context above. If the answer is not present, say "Please consult the INFINITUM 2025 organizers."
    # """
    prompt = f"""
    Context: {context}

    Question: {query}

    Answer ONLY using the context above. If the answer is not present, say "Please consult the INFINITUM 2025 organizers."
    """

    response = llm_pipeline(
        prompt,
        max_length = max_length,
        # num_beams = num_beams,
        # temperature = 0.7,
        num_return_sequences = 1
    )

    return response[0]['generated_text']

@app.route('/api/init', methods=['POST'])
def initialise():
    global llm_pipeline, model, index
    try:
        if model is None or index is None:
            load_model_and_index()
        
        llm_pipeline = initialize_llm()
        return jsonify({"message": "Model loaded successfully"}), 200
    except Exception as e:
        return jsonify({"startup error": str(e)}), 500

@app.route('/api/sendprompt', methods = ['POST'])
def generate():
    try:
        data = request.get_json()
        user_input  = (data.get('text') or '').strip()
        if not user_input:
            return jsonify({"error": "No input message provided"}), 400

        candidates = hybrid_retrieve(user_input)
        if not candidates:
            return jsonify({"reponse": "I'm sorry I cannot answer that"}), 200

        best_doc = rerank(user_input, candidates)
        query_emb = model.encode([user_input])
        doc_emb = model.encode([best_doc])
        sim_score = cosine_similarity(query_emb, doc_emb)[0][0]

        if sim_score < relevance_threshold:
            return jsonify({"response": "I'm sorry, no relevant answer found"}), 200

        



        # expansion_prompt = f"Generate 3 query variations for: {user_input}"
        # expansions = llm_pipeline(expansion_prompt, max_length = 100, num_return_sequences = 1)
        # expansion_text = expansions[0]['generated_text']

        # expanded_queries = [user_input] + [q.strip() for q in expansion_text.split('\n') if q.strip()]

        # all_relevant_docs = []
        # for query in expanded_queries:
        #     query_embedding = model.encode([query])
        #     distances, indices = index.search(query_embedding, k = 3)
        #     docs = [lines[i] for i in indices[0]]
        #     all_relevant_docs.extend(docs)

        # all_relevant_docs = list(set(all_relevant_docs))

        # relevant_doc_embeddings = model.encode(all_relevant_docs)
        # query_embedding = model.encode([user_input])
        # similarity_scores = cosine_similarity(query_embedding, relevant_doc_embeddings).flatten()

        # max_score = max(similarity_scores) if len(similarity_scores) > 0 else 0

        # if max_score < relevance_threshold:
        #     return jsonify({"response": "I'm sorry, I cannot answer that."})
        
        # # best_doc = all_relevant_docs[similarity_scores.argmax()]
        # best_doc = all_relevant_docs[int(np.argmax(similarity_scores))]

        # context = "Relevant information:\n" + best_doc

        # answer  = generate_answer(user_input, context, llm_pipeline)
        answer  = generate_answer(user_input, best_doc, llm_pipeline)
        return jsonify({
            "answer":answer,
            "context":best_doc
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port = int(os.getenv("backend-port", 5000)))  # Use environment variable or default to 5000

