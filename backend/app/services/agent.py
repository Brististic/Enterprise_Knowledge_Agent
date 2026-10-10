import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class KnowledgeAgent:
    def __init__(self, store, reranker, embedder):
        self.store = store
        self.reranker = reranker
        self.embedder = embedder

        # Initialize OpenAI-compatible client
        api_key = os.getenv("LLM_API_KEY", "dummy-key")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _generate_grounded_answer(self, query: str, context_blocks: List[str]) -> str:
        """Calls the LLM with grounded context and citation instructions."""
        formatted_context = "\n\n".join(context_blocks)
        
        system_prompt = (
            "You are an enterprise AI document intelligence assistant. "
            "Answer the user query solely using the provided context chunks. "
            "Every claim, statistic, or fact you state MUST include its inline citation key "
            "(e.g., [cite-1], [cite-2]) directly corresponding to the context chunk used. "
            "Do not hallucinate facts outside the context. If the context does not contain "
            "the answer, state clearly that the document does not mention it."
        )

        user_prompt = f"Context Documents:\n{formatted_context}\n\nUser Question: {query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with LLM synthesis provider: {str(e)}"

    def answer_query(self, query: str) -> Dict[str, Any]:
        print(f"\n[Agent] Received query: '{query}'")

        # 1. First-Stage Retrieval (Hybrid Dense + BM25 via RRF)
        query_vec = self.embedder.embed_query(query)
        candidates = self.store.search(query, query_vec, top_k=10)

        # 2. Second-Stage Neural Re-ranking (Cross-Encoder)
        top_chunks = self.reranker.rerank(query, candidates, top_k=3)

        # 3. Context Sufficiency Check
        if not top_chunks or top_chunks[0]["cross_score"] < -6.0:
            return {
                "query": query,
                "answer": "I could not find sufficient relevant evidence in the document to answer this question.",
                "citations": []
            }

        # 4. Prepare Citation Metadata and Context Prompts
        citations = []
        context_blocks = []
        for idx, chunk in enumerate(top_chunks):
            citation_id = f"cite-{idx + 1}"
            citations.append({
                "id": citation_id,
                "page": chunk["page"],
                "bbox": chunk["bbox"],
                "content_preview": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
                "confidence": chunk["cross_score"]
            })
            context_blocks.append(f"[{citation_id}] (Page {chunk['page']}):\n{chunk['content']}")

        # 5. Natural Language LLM Synthesis
        synthesized_answer = self._generate_grounded_answer(query, context_blocks)

        return {
            "query": query,
            "answer": synthesized_answer,
            "citations": citations
        }