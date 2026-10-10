import time
from app.services.parser import MultimodalDocumentParser
from app.services.embedder import EmbeddingService
from app.services.vector_store import HybridStore
from app.services.reranker import NeuralReranker
import numpy as np

def run_benchmark():
    print("--- Initializing Evaluation Benchmark ---")
    parser = MultimodalDocumentParser()
    embedder = EmbeddingService()
    store = HybridStore()
    reranker = NeuralReranker()

    pdf_path = "data/uploads/sample.pdf"
    chunks = parser.parse_pdf(pdf_path, max_pages=2)
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(chunks, embeddings)

    # Labeled ground-truth benchmark test cases
    test_suite = [
        {
            "query": "What is scheduled for Day 5?",
            "expected_keyword": "Day 5"
        },
        {
            "query": "What is scheduled for Day 12?",
            "expected_keyword": "Day 12"
        },
        {
            "query": "Dockerization for Machine Learning APIs",
            "expected_keyword": "Day 6"
        },
        {
            "query": "Background Tasks and Rate Limiting",
            "expected_keyword": "Day 4"
        }
    ]

    print(f"\nRunning comparative benchmark over {len(test_suite)} queries...\n")

    dense_mrr = []
    hybrid_mrr = []
    rerank_mrr = []

    dense_p3 = []
    hybrid_p3 = []
    rerank_p3 = []

    latencies_ms = []

    for item in test_suite:
        q = item["query"]
        target = item["expected_keyword"].lower()
        t0 = time.perf_counter()

        # 1. Baseline: Dense Vector Search Only
        q_vec = embedder.embed_query(q)
        q_arr = np.array(q_vec, dtype=np.float32)
        dense_scores = [float(np.dot(q_arr, doc_emb)) for doc_emb in store.embeddings]
        dense_indices = np.argsort(dense_scores)[::-1][:10]
        dense_results = [store.chunks[i]["content"].lower() for i in dense_indices]

        # 2. Stage 1: Hybrid Search (RRF)
        hybrid_results_raw = store.search(q, q_vec, top_k=10)
        hybrid_results = [c["content"].lower() for c in hybrid_results_raw]

        # 3. Stage 2: Hybrid + Cross-Encoder Re-ranking
        reranked_raw = reranker.rerank(q, hybrid_results_raw, top_k=3)
        reranked_results = [c["content"].lower() for c in reranked_raw]

        latencies_ms.append((time.perf_counter() - t0) * 1000)

        # Helper metrics
        def get_rr(results):
            for rank, text in enumerate(results):
                if target in text:
                    return 1.0 / (rank + 1)
            return 0.0

        def get_p3(results):
            matches = sum(1 for text in results[:3] if target in text)
            return matches / 3.0

        dense_mrr.append(get_rr(dense_results))
        hybrid_mrr.append(get_rr(hybrid_results))
        rerank_mrr.append(get_rr(reranked_results))

        dense_p3.append(get_p3(dense_results))
        hybrid_p3.append(get_p3(hybrid_results))
        rerank_p3.append(get_p3(reranked_results))

    print("=" * 60)
    print("           RETRIEVAL ABLATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Method':<32} | {'MRR':<8} | {'Precision@3':<12}")
    print("-" * 60)
    print(f"{'1. Baseline (Dense Vector Only)':<32} | {np.mean(dense_mrr):.3f}    | {np.mean(dense_p3):.3f}")
    print(f"{'2. Hybrid (BGE + BM25 via RRF)':<32} | {np.mean(hybrid_mrr):.3f}    | {np.mean(hybrid_p3):.3f}")
    print(f"{'3. Hybrid + Cross-Encoder':<32} | {np.mean(rerank_mrr):.3f}    | {np.mean(rerank_p3):.3f}")
    print("=" * 60)
    print(f"Average Pipeline Latency: {np.mean(latencies_ms):.1f} ms / query\n")

if __name__ == "__main__":
    run_benchmark()