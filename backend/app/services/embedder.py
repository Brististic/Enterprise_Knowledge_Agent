from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device="cpu")
        print("Embedding model loaded.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates 384-dimensional dense vectors for a list of text strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query."""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()