from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )


embedding_model = EmbeddingModel()
