import os
import pickle
import re

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from rag.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    VECTOR_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    BM25_PATH,
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
)


class HybridRetriever:
    def __init__(self):
        self.embedding_model_name = (
            EMBEDDING_MODEL
        )
        self.reranker_model_name = (
            RERANKER_MODEL
        )

        self.vector_top_k = (
            VECTOR_TOP_K
        )
        self.bm25_top_k = (
            BM25_TOP_K
        )
        self.final_top_k = (
            FINAL_TOP_K
        )

        self.chunk_tokens = (
            CHUNK_TOKENS
        )
        self.chunk_overlap = (
            CHUNK_OVERLAP
        )

        self.embedder = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )
        self.embedder.max_seq_length = 512

        self.reranker = (
            CrossEncoder(
                RERANKER_MODEL
            )
        )

        self.client = QdrantClient(
            url=(
                f"http://"
                f"{QDRANT_HOST}:"
                f"{QDRANT_PORT}"
            )
        )

        if not self.client.collection_exists(
            COLLECTION_NAME
        ):
            self.client.create_collection(
                collection_name=(
                    COLLECTION_NAME
                ),
                vectors_config=(
                    VectorParams(
                        size=(
                            self.embedder
                            .get_sentence_embedding_dimension()
                        ),
                        distance=Distance.COSINE,
                        hnsw_config=(
                            HnswConfigDiff(
                                m=16,
                                ef_construct=100,
                            )
                        ),
                    )
                ),
            )

        self.records = []
        self.bm25 = None
        self._load_bm25()

    def _load_bm25(self):
        if not os.path.exists(
            BM25_PATH
        ):
            return

        try:
            with open(
                BM25_PATH,
                "rb",
            ) as f:
                self.records = pickle.load(
                    f
                )

            self._rebuild_bm25()

        except Exception:
            self.records = []
            self.bm25 = None

    def _save_bm25(self):
        os.makedirs(
            os.path.dirname(
                BM25_PATH
            ),
            exist_ok=True,
        )

        with open(
            BM25_PATH,
            "wb",
        ) as f:
            pickle.dump(
                self.records,
                f,
            )

    def _tokens(self, text):
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _rebuild_bm25(self):
        if not self.records:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(
            [
                self._tokens(
                    r["text"]
                )
                for r in self.records
            ]
        )

    def document_exists(
        self,
        document_id,
    ):
        points, _ = (
            self.client.scroll(
                collection_name=(
                    COLLECTION_NAME
                ),
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(
                                value=document_id
                            ),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        )

        return bool(points)

    def upsert_points(
        self,
        points,
    ):
        for start in range(
            0,
            len(points),
            100,
        ):
            self.client.upsert(
                collection_name=(
                    COLLECTION_NAME
                ),
                points=points[
                    start:start + 100
                ],
                wait=True,
            )

    def add_bm25_records(
        self,
        records,
    ):
        self.records.extend(
            records
        )

        self._rebuild_bm25()
        self._save_bm25()

    def _vector_search(
        self,
        query,
    ):
        vector = (
            self.embedder.encode(
                query,
                normalize_embeddings=True,
            ).tolist()
        )

        result = (
            self.client.query_points(
                collection_name=(
                    COLLECTION_NAME
                ),
                query=vector,
                limit=self.vector_top_k,
                with_payload=True,
                with_vectors=False,
            )
        )

        return result.points

    def _bm25_search(
        self,
        query,
    ):
        if not self.bm25:
            return []

        scores = (
            self.bm25.get_scores(
                self._tokens(query)
            )
        )

        indices = sorted(
            range(
                len(scores)
            ),
            key=lambda i: scores[i],
            reverse=True,
        )[:self.bm25_top_k]

        return [
            {
                "payload": self.records[i],
                "score": float(
                    scores[i]
                ),
            }
            for i in indices
            if scores[i] > 0
        ]

    def search(
        self,
        query,
    ):
        vector_results = (
            self._vector_search(
                query
            )
        )

        bm25_results = (
            self._bm25_search(
                query
            )
        )

        candidates = {}

        for point in vector_results:
            payload = point.payload

            candidates[
                payload["chunk_id"]
            ] = {
                "payload": payload,
                "vector_score": float(
                    point.score
                ),
                "bm25_score": 0.0,
            }

        for result in bm25_results:
            payload = result[
                "payload"
            ]

            key = payload[
                "chunk_id"
            ]

            if key not in candidates:
                candidates[key] = {
                    "payload": payload,
                    "vector_score": 0.0,
                    "bm25_score": result[
                        "score"
                    ],
                }
            else:
                candidates[key][
                    "bm25_score"
                ] = result[
                    "score"
                ]

        candidates = list(
            candidates.values()
        )

        if not candidates:
            return [], {
                "vector_candidates": 0,
                "bm25_candidates": 0,
                "hybrid_candidates": 0,
            }

        pairs = [
            (
                query,
                item["payload"]["text"],
            )
            for item in candidates
        ]

        rerank_scores = (
            self.reranker.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        )

        for item, score in zip(
            candidates,
            rerank_scores,
        ):
            item[
                "rerank_score"
            ] = float(score)

        candidates.sort(
            key=lambda item: item[
                "rerank_score"
            ],
            reverse=True,
        )

        return (
            candidates[:self.final_top_k],
            {
                "vector_candidates": len(
                    vector_results
                ),
                "bm25_candidates": len(
                    bm25_results
                ),
                "hybrid_candidates": len(
                    candidates
                ),
            },
        )

    def build_context(
        self,
        results,
    ):
        parts = []

        for i, item in enumerate(
            results,
            1,
        ):
            p = item["payload"]

            parts.append(
                f"[SOURCE {i}]\n"
                f"PDF: {p['source']}\n"
                f"Page: {p['page']}\n"
                f"Type: {p['content_type']}\n"
                f"Text:\n{p['text']}"
            )

        return "\n\n".join(
            parts
        )

    def stats(self):
        try:
            info = (
                self.client
                .get_collection(
                    COLLECTION_NAME
                )
            )

            points = (
                info.points_count or 0
            )

            documents = {
                r["document_id"]
                for r in self.records
            }

            pages = {
                (
                    r["document_id"],
                    r["page"],
                )
                for r in self.records
            }

            return {
                "documents": len(
                    documents
                ),
                "pages": len(pages),
                "chunks": points,
            }

        except Exception:
            return {
                "documents": 0,
                "pages": 0,
                "chunks": 0,
            }
