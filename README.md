# Multimodal Enterprise Knowledge Agent

An enterprise-grade document intelligence platform featuring vision-layout parsing, two-stage hybrid retrieval, neural re-ranking, and dynamic visual citation grounding.

## Architecture

1. **Vision-Layout Document Ingestion**: PyMuPDF & Microsoft Florence-2 extract document blocks, tables, and normalized spatial bounding boxes (`[x1, y1, x2, y2]`).
2. **Two-Stage Hybrid Search**:
   - **Dense Embedding**: `BAAI/bge-small-en-v1.5` for semantic representation.
   - **Sparse Lexical Index**: BM25 keyword matching.
   - **Fusion**: Reciprocal Rank Fusion (RRF) to score top candidate chunks.
3. **Neural Re-Ranking**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) computes full joint attention over query-passage pairs.
4. **Natural Language Synthesis**: Ollama / Llama-3.2 generating structured answers with strict inline citations (`[cite-1]`, `[cite-2]`).
5. **Visual Grounding UI**: Next.js & Tailwind CSS split-screen layout dynamically highlighting source bounding boxes on the document page.

## Getting Started

### Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
python -m app.main

### Frontend Setup
```bash
cd frontend
npm install
npm run dev

