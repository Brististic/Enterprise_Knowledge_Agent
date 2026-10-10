from app.services.parser import MultimodalDocumentParser
from app.services.embedder import EmbeddingService
from app.services.vector_store import HybridStore
import json

def run():
    parser = MultimodalDocumentParser()
    embedder = EmbeddingService()
    store = HybridStore()

    sample_pdf = "data/uploads/sample.pdf"
    print(f"\n--- 1. Parsing Document: {sample_pdf} ---")
    chunks = parser.parse_pdf(sample_pdf, max_pages=2)
    print(f"Extracted {len(chunks)} chunks.")

    print("\n--- 2. Computing Dense Embeddings ---")
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    print("\n--- 3. Indexing Chunks in Hybrid Store ---")
    store.add_chunks(chunks, embeddings)
    print("Indexing complete.")

    # 4. Run a query search
    query = "revenue growth or technical summary"
    print(f"\n--- 4. Querying: '{query}' ---")
    q_emb = embedder.embed_query(query)
    matches = store.search(query, q_emb, top_k=2)

    print("\nSearch Results (Retrieved Chunks with Bounding Boxes):")
    print(json.dumps(matches, indent=2))

if __name__ == "__main__":
    run()