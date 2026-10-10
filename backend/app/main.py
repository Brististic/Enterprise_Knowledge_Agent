import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.parser import MultimodalDocumentParser
from app.services.embedder import EmbeddingService
from app.services.vector_store import HybridStore
from app.services.reranker import NeuralReranker
from app.services.agent import KnowledgeAgent

app = FastAPI(title="Multimodal Enterprise Knowledge Agent API")

# Enable CORS for Next.js / React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Singletons initialized in memory
print("Initializing Agent pipeline services...")
parser = MultimodalDocumentParser()
embedder = EmbeddingService()
store = HybridStore()
reranker = NeuralReranker()
agent = KnowledgeAgent(store=store, reranker=reranker, embedder=embedder)
print("All pipeline services ready.")


class QueryRequest(BaseModel):
    query: str


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, extracts multimodal chunks, embeds, and indexes them."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingestion
    chunks = parser.parse_pdf(file_path, max_pages=10)
    if not chunks:
        raise HTTPException(status_code=422, detail="No readable text or visual regions found.")

    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(chunks, embeddings)

    return {
        "status": "success",
        "filename": file.filename,
        "indexed_chunks": len(chunks)
    }


@app.post("/api/query")
async def query_knowledge_base(payload: QueryRequest):
    """Queries the knowledge base and returns answer with grounded visual citations."""
    if not store.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    result = agent.answer_query(payload.query)
    return result


@app.get("/api/document/{filename}")
async def get_document(filename: str):
    """Serves the raw PDF to the frontend document viewer."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "indexed_chunks": len(store.chunks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


from fastapi.responses import Response

@app.get("/api/document/{filename}/page/{page_number}")
async def get_page_preview(filename: str, page_number: int):
    """Renders and returns a specific PDF page as a PNG image for visual grounding."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    import pymupdf as fitz
    doc = fitz.open(file_path)
    if page_number < 1 or page_number > len(doc):
        doc.close()
        raise HTTPException(status_code=400, detail="Page number out of bounds")

    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    doc.close()

    return Response(content=img_bytes, media_type="image/png")