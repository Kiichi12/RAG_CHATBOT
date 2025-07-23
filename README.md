# RAG_CHATBOT

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about **INFINITUM 2025**, a major school quiz competition focused on logical reasoning and mathematical problem-solving. This project combines a React frontend with a Python (Flask) backend utilizing modern NLP models for accurate, context-driven responses.

---

## Features

- **Chatbot Interface:** Friendly web-based chatbot that answers queries about the quiz event.
- **Retrieval-Augmented Generation:** Uses advanced retrieval (BM25, embedding, reranking) to source information from official documents and generate relevant answers.
- **Multi-Round Quiz Details:** Provides information on quiz structure, eligibility, prizes, registration, and history.
- **Separation by Category:** Handles queries for both Junior (Classes 7–10) and Senior (Classes 11–12) participants.
- **Efficient & Memory-Conscious:** Designed to run on limited hardware, using optimized models and routines.

---

## Project Structure

```
RAG_CHATBOT/
├── backend/              # Python Flask backend API
│   ├── app2.py
│   ├── app3.py
│   ├── quiz_details.txt  # Main knowledge base
│   └── quiz2.txt         # Additional quiz info
├── src/                  # React frontend
│   └── App.jsx
├── index.html
└── README.md
```

---

## How It Works

1. **User Message:** User asks a question in the chat UI.
2. **Backend Processing:**
   - Initializes NLP models (retriever, reranker, LLM).
   - Retrieves relevant context passages from `quiz_details.txt`.
   - Generates a concise answer using Hugging Face models.
3. **Frontend Display:** User sees the bot's answer in real time.

---

## Quickstart

### Backend

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Set environment variable for Hugging Face API token (if required):**
   ```bash
   export HF_API_TOKEN=your_hf_token
   ```
3. **Run the Flask app:**
   ```bash
   python app3.py
   ```

### Frontend

1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Start the frontend:**
   ```bash
   npm run dev
   ```

---

## Technologies Used

- **Frontend:** React, Vite, Axios
- **Backend:** Python, Flask, transformers, sentence-transformers, rank_bm25, FAISS, NLTK
- **NLP Models:** Google FLAN-T5 (for generation), MiniLM and TinyBERT (for retrieval/reranking)

---

## Sample Questions

- "What are the eligibility criteria for the quiz?"
- "When is the Grand Finale held?"
- "What are the prizes for top teams?"
- "How is the competition structured?"

---

## Acknowledgements

- Hugging Face for open-source models
- NIT Calicut and INFINITUM organizers

---

## License

This project is for academic and event information purposes only. See individual file headers for third-party licenses.
