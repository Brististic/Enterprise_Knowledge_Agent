import json
from app.services.parser import MultimodalDocumentParser
from app.services.embedder import EmbeddingService
from app.services.vector_store import HybridStore
from app.services.reranker import NeuralReranker
from app.services.agent import KnowledgeAgent

def main():
    # 1. Initialize components
    parser = MultimodalDocumentParser()
    embedder = EmbeddingService()
    store = HybridStore()
    reranker = NeuralReranker()

    # 2. Ingest document
    pdf_path = "data/uploads/sample.pdf"
    chunks = parser.parse_pdf(pdf_path, max_pages=2)
    embeddings = embedder.embed_texts([c["content"] for c in chunks])
    store.add_chunks(chunks, embeddings)

    # 3. Instantiate Agent
    agent = KnowledgeAgent(store=store, reranker=reranker, embedder=embedder)

    # 4. Ask a test question based on your document content
    test_query = "What is scheduled for Day 12?"
    result = agent.answer_query(test_query)

    print("\n--- Final Grounded Agent Response ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()