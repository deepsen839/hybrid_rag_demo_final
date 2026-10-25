import os
import time
import streamlit as st

from rag.ingestion import PDFIngestor
from rag.retrieval import HybridRetriever
from rag.generation import generate_stream


st.set_page_config(
    page_title="Large PDF Hybrid RAG Demo",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def load_system():
    retriever = HybridRetriever()
    ingestor = PDFIngestor()
    return ingestor, retriever

ingestor, retriever = load_system()

# -----------------------------
# Header
# -----------------------------
st.title("📚 Large PDF Hybrid RAG Chatbot")
st.caption(
    "PyMuPDF + Tesseract OCR + BGE embeddings + Qdrant HNSW "
    "+ BM25 + CrossEncoder + Groq"
)

# -----------------------------
# Sidebar: ingestion
# -----------------------------
with st.sidebar:
    st.header("📥 Document Ingestion")

    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Demo target: 10+ PDFs, each 200+ pages.",
    )

    if files:
        st.info(f"{len(files)} PDF(s) selected")

    if st.button("🚀 Ingest Documents", use_container_width=True):
        if not files:
            st.warning("Select PDF files first.")
        else:
            progress = st.progress(0)
            total_pages = 0
            total_chunks = 0

            for i, file in enumerate(files):
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        pages, chunks = ingestor.index_pdf(
                            file,
                            retriever,
                        )
                        total_pages += pages
                        total_chunks += chunks

                        if chunks == 0:
                            st.warning(
                                f"{file.name}: already indexed or no extractable text."
                            )
                        else:
                            st.success(
                                f"{file.name}: {pages} pages → {chunks} chunks"
                            )
                    except Exception as exc:
                        st.error(f"{file.name}: {exc}")

                progress.progress((i + 1) / len(files))

            st.success(
                f"Indexed {total_chunks:,} chunks from "
                f"{total_pages:,} pages."
            )

    st.divider()

    st.header("⚙️ RAG Configuration")
    st.write(f"Embedding: `{retriever.embedding_model_name}`")
    st.write(f"Reranker: `{retriever.reranker_model_name}`")
    st.write(f"Vector Top-K: `{retriever.vector_top_k}`")
    st.write(f"BM25 Top-K: `{retriever.bm25_top_k}`")
    st.write(f"Final Top-K: `{retriever.final_top_k}`")
    st.write(f"Chunk size: `{retriever.chunk_tokens}` tokens")
    st.write(f"Overlap: `{retriever.chunk_overlap}` tokens")

# -----------------------------
# Demo status
# -----------------------------
stats = retriever.stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Indexed PDFs", stats["documents"])
c2.metric("Indexed Pages", stats["pages"])
c3.metric("Indexed Chunks", stats["chunks"])
c4.metric("Vector DB", "Qdrant HNSW")

if stats["documents"] < 10:
    st.warning(
        "Demo target: ingest at least 10 PDFs. "
        "The application is ready for 10+ PDFs; the metric above shows "
        "the number actually indexed."
    )

# -----------------------------
# Ingestion architecture
# -----------------------------
with st.expander("🔧 Ingestion Pipeline", expanded=False):
    st.markdown(
        """
        **PDF → PyMuPDF native extraction → OCR fallback → cleaning →
        language detection → header/footer removal → token chunking →
        open-source embeddings → Qdrant HNSW + BM25**
        """
    )

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask a question about the PDF knowledge base..."
)

if query:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not configured.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # -----------------------------
    # Retrieval
    # -----------------------------
    retrieval_start = time.perf_counter()

    with st.spinner("🔎 Searching the PDF knowledge base..."):
        results, retrieval_metrics = retriever.search(query)

    retrieval_time = time.perf_counter() - retrieval_start

    if not results:
        answer = "Context not found in documents."

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.stop()

    # -----------------------------
    # Retrieval visualization
    # -----------------------------
    with st.expander(
        f"🔍 Retrieved Chunks ({len(results)})",
        expanded=True,
    ):
        st.caption(
            f"Retrieval + reranking: {retrieval_time:.2f}s"
        )

        for rank, item in enumerate(results, 1):
            p = item["payload"]

            st.markdown(
                f"**#{rank} — {p['source']} — "
                f"Page {p['page']}**"
            )

            score_cols = st.columns(3)
            score_cols[0].metric(
                "Rerank",
                f"{item['rerank_score']:.3f}",
            )
            score_cols[1].metric(
                "Vector",
                f"{item.get('vector_score', 0):.3f}",
            )
            score_cols[2].metric(
                "BM25",
                f"{item.get('bm25_score', 0):.3f}",
            )

            st.code(
                p["text"][:1200],
                language="text",
            )

            st.caption(
                f"Language: {p.get('language', 'unknown')} · "
                f"Type: {p.get('content_type', 'unknown')} · "
                f"Chunk: {p.get('chunk_index', '-')}"
            )

            st.divider()

    context = retriever.build_context(results)

    # -----------------------------
    # Generation
    # -----------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        llm_start = time.perf_counter()

        try:
            for token in generate_stream(query, context):
                answer += token
                placeholder.markdown(answer + "▌")

            llm_time = time.perf_counter() - llm_start

            # Build source list from retrieved chunks
            source_lines = []
            seen = set()

            for item in results:
                p = item["payload"]
                key = (p["source"], p["page"])

                if key not in seen:
                    seen.add(key)
                    source_lines.append(
                        f"- **{p['source']}** — page {p['page']}"
                    )

            answer += (
                "\n\n---\n"
                "**📚 Sources**\n\n"
                + "\n".join(source_lines)
            )

            placeholder.markdown(answer)

            total_time = retrieval_time + llm_time

            if total_time <= 5:
                status = "🟢 Within 2–5s target"
            else:
                status = "🟡 Above 5s target"

            st.caption(
                f"{status} · "
                f"Retrieval: {retrieval_time:.2f}s · "
                f"LLM: {llm_time:.2f}s · "
                f"End-to-end: {total_time:.2f}s"
            )

            with st.expander("📊 Query Metrics"):
                st.json({
                    "retrieval_seconds": round(retrieval_time, 3),
                    "llm_seconds": round(llm_time, 3),
                    "end_to_end_seconds": round(total_time, 3),
                    **retrieval_metrics,
                })

        except Exception as exc:
            st.error(f"Generation error: {exc}")
            answer = f"Generation error: {exc}"

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
