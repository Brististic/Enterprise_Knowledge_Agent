from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class NeuralReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder Reranker: {model_name} on cpu...")
        self.model = CrossEncoder(model_name, device="cpu")
        print("Reranker loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Prepare (query, passage) pairs
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)

        # Attach cross-encoder scores
        for idx, score in enumerate(scores):
            candidates[idx]["cross_score"] = float(round(score, 4))

        # Sort by neural cross-attention score descending
        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]