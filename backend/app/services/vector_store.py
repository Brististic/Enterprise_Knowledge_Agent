import numpy as np
from typing import List, Dict, Any
import math
import re

class HybridStore:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.corpus_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        for emb in embeddings:
            self.embeddings.append(np.array(emb, dtype=np.float32))

        # Build BM25 index components
        self.corpus_tokens = [self._tokenize(c["content"]) for c in self.chunks]
        self.avg_doc_len = sum(len(doc) for doc in self.corpus_tokens) / max(len(self.corpus_tokens), 1)

        self.doc_freqs = {}
        for doc in self.corpus_tokens:
            for token in set(doc):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _score_bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        query_tokens = self._tokenize(query)
        num_docs = len(self.chunks)
        scores = [0.0] * num_docs

        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            for idx, doc in enumerate(self.corpus_tokens):
                tf = doc.count(token)
                denom = tf + k1 * (1.0 - b + b * (len(doc) / self.avg_doc_len))
                scores[idx] += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-6))
        return scores

    def search(self, query: str, query_embedding: List[float], top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Performs Hybrid Search using Reciprocal Rank Fusion (RRF)."""
        if not self.chunks:
            return []

        # 1. Dense search ranking (Cosine Similarity)
        q_emb = np.array(query_embedding, dtype=np.float32)
        dense_scores = [float(np.dot(q_emb, doc_emb)) for doc_emb in self.embeddings]
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # 2. Sparse lexical ranking (BM25)
        bm25_scores = self._score_bm25(query)
        sparse_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {i: 0.0 for i in range(len(self.chunks))}
        for rank, idx in enumerate(dense_ranked_indices):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_ranked_indices):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # Sort combined results
        final_ranking = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for idx, score in final_ranking:
            item = dict(self.chunks[idx])
            item["rrf_score"] = round(score, 5)
            results.append(item)

        return results