from sentence_transformers import SentenceTransformer
from rag.config import EMBEDDING_MODEL


class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512

    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
