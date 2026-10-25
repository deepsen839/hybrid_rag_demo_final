# Large PDF Hybrid RAG — Live Demo

This implementation is designed for the stated deliverable:

- 10+ large PDFs, including 200+ page PDFs
- open-source embeddings
- real-time RAG chat
- Qdrant HNSW ANN search
- BM25 lexical retrieval
- CrossEncoder reranking
- source/page provenance
- ingestion visualization
- retrieved-chunk visualization
- latency measurement
- Recall@K and MRR evaluation
- Docker Compose deployment

## Architecture

PDF
-> PyMuPDF native extraction
-> Tesseract OCR fallback
-> language detection
-> repeated header/footer filtering
-> deterministic token chunks
-> BGE-small embeddings
-> Qdrant HNSW
+
BM25
-> hybrid candidate set
-> CrossEncoder
-> top 5
-> Groq
-> answer + PDF/page citations

## Important chunking decision

BGE-small-en-v1.5 has a 512-token model context. The implementation
uses 450-token chunks with 75-token overlap so the embedding input
does not exceed the model context.

## Start

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY

docker compose build
docker compose up -d
```

Open:

http://localhost:8502

## Monitor

```bash
docker compose ps
docker compose logs -f streamlit-app
```

## Evaluation

First ingest the evaluation corpus, then:

```bash
docker compose exec streamlit-app python -m rag.evaluation
```

Replace `evaluation/test_questions.json` with questions and known
relevant pages from the actual corpus.

## 2–5 second requirement

The UI reports:

- retrieval + reranking latency
- LLM latency
- end-to-end latency

The 2–5 second target is a benchmark requirement, not a guaranteed
property of every hardware/API environment. For the demo, benchmark
with the actual 10+ PDF corpus and record p95 latency.

## Demo checklist

1. Ingest at least 10 PDFs.
2. Verify the sidebar shows 10+ Indexed PDFs.
3. Ask a question.
4. Expand "Retrieved Chunks".
5. Show vector/BM25/reranker scores.
6. Show the generated answer.
7. Show PDF filename + page citations.
8. Show end-to-end latency.
9. Run evaluation and record Recall@5, MRR and latency.
