# Healthcare RAG Assistant RFC

## Summary
A Retrieval-Augmented Generation (RAG) system designed for extraction and QA over clinical and regulatory documents (PDF/HTML/Excel/CSV). The solution is privacy-aware, capable of metadata-driven chunking and validation, and offers factual LLM answers for healthcare professionals and regulatory users. It demonstrates student-level system design and product thinking.

## Purpose & Users
- Rapid information retrieval for clinicians, regulatory specialists, and healthcare researchers.
- Support for mixed-document formats with privacy constraints.

## Architectural Constraints
- No secrets tracked in repo; client-side processing recommended for private data.
- Factual QA prioritized (low LLM temperature).
- Modular, extensible pipeline.

## Pipeline Steps
1. **Ingest/Prepare**: Accept PDF, HTML, or Excel/CSV sources. Parse documents using pdfplumber/fitz and pandas.
2. **Data Processing**: Extract text in document-level and chunk by RecursiveCharacterTextSplitter.
3. **Generate Embeddings**: Encode text chunks using OpenAIEmbeddings; process metadata (filename, chunk_index, source).
4. **Save Embeddings to Vector DB**: Store using FAISS, assign metadata, validate chunk content, and ensure privacy measures.
5. **Query Handling**: RAG pipeline fetches relevant chunks from Vector DB, constructs prompt, and assembles context.
6. **Prompt & LLM Response**: Compose strict context-only prompts and query LLM (ChatOpenAI) at low temperature.
7. **User Response/UI**: Streamlit UI provides transparent matching of answers and source, including similarity scores.

## Key Product/Architecture Decisions
- **chunk_size**: 1000 tokens (default); overlap: 200 tokens. Enables context retention and granular search.
- **Metadata Keys**: filename, chunk_index, document_type, source, receipt_time. Transparency for provenance checks.
- **Privacy/Client-Side Processing**: Recommend local chunking if strict privacy required. Remove sensitive info before vectorization.
- **Low LLM Temperature**: Factual answers prioritized using temperature ≈ 0.1-0.2.
- **Validation & QA**: Validate text chunks and metadata for source integrity. Include metadata in RAG prompt for traceability.

## Example: Metadata Snippet
```json
{
  "filename": "clinical_trial_protocol.pdf",
  "chunk_index": 17,
  "source": "uploaded_by_client",
  "document_type": "protocol",
  "receipt_time": "2025-11-09T12:30Z"
}
```

## Future Improvements
- Multi-user dashboards and access logging.
- Add optional encrypted client-side vector DB.
- Extend QA to support multi-document synthesis and summaries.
- Add document upload directly from regulatory platforms/APIs.
- LLM validation against reference regulatory texts (FDA, EMA, CDSCO).
- Automated privacy audits for client-uploaded documents.

---
