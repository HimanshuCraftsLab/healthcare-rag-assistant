# Healthcare RAG Assistant (Clinical & Regulatory Documents)

A Retrieval-Augmented Generation (RAG) system to extract, index, and provide factual answers from clinical and regulatory documents (PDF/HTML/CSV/Excel). Built as a student project demonstrating system design, product thinking, and end-to-end implementation.

![Architecture Diagram](architecture-diagram.png)

![Uploading image.png…]()


## 🔍 Overview
- Upload multi-format documents (PDF, HTML, TXT, CSV, XLSX).
- Extract and chunk text, generate embeddings, store in FAISS, and perform semantic retrieval.
- Build context-aware prompts and generate accurate LLM responses with low temperature for factualness.
- Streamlit UI for queries and matched-source transparency.

## 🧭 Architecture (high level)
- Client upload → parsing (pdfplumber / fitz) → chunking (RecursiveCharacterTextSplitter) → embeddings (OpenAIEmbeddings) → FAISS vector store → similarity search → prompt assembly → LLM (ChatOpenAI) → response.
- Key design choices: chunk_size=1000 tokens, overlap=200 tokens, metadata: filename, chunk_index, source; low LLM temperature (~0.1) for factual answers; privacy-aware processing.

## ⭐ Key product decisions
- Document summaries and suggested questions to reduce user cognitive load.
- Traceable metadata for each chunk to allow source verification.
- Client-side processing recommendation for privacy-sensitive workflows.
- RFC-style documentation included for architecture and reasoning.

## 🛠️ Features
- Multi-format ingestion: pdf, txt, html, csv, xlsx
- Text extraction + chunking with metadata
- FAISS-based vector store and similarity search
- RAG prompt construction with strict context-only answers
- Streamlit UI showing response + matched source text and scores

## ▶️ How to run (local)
```bash
git clone https://github.com/<your-username>/healthcare-rag-assistant.git
cd healthcare-rag-assistant
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
STREAMLIT_RUN_COMMAND: streamlit run app/healthcare_rag_app_main.py
```

📁 Code structure
app/healthcare_rag_app_main.py — main Streamlit app
architecture-diagram.png — system diagram
rfc-notes.md — RFC-style notes and decisions

ℹ️ Notes & Caution
Do not commit any .env or API keys. Remove .env before pushing. If you find an API key in the attached .env, please notify me to rotate it.
This repository is meant as a demonstration. For productionization, see TODOs in rfc-notes.md.

📄 License
MIT License — see LICENSE file.

🔗 Links
LinkedIn: <YOUR_LINKEDIN_URL>
GitHub profile: <YOUR_GITHUB_URL>

*(Replace `<your-username>`, `<YOUR_LINKEDIN_URL>`, and `<YOUR_GITHUB_URL>` with the appropriate values; ask me if you need.)*
