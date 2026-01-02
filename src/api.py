"""FastAPI wrapper for the Multimodal RAG project.

Provides endpoints to ingest documents and query the RAG pipeline.
"""
from typing import Optional, List
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import main as ingest_main
from src.config import LLM, VECTOR_STORE_PATH, IMAGE_STORE_PATH
from src.vector_store_utils import load_vector_store
from src.rag_pipeline import multimodal_rag_pipeline, retrieval_multimodal

app = FastAPI(title="Multimodal RAG API")

# In-memory handles for vector store and image data
vector_store = None
image_data_store = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_images: bool = False


class IngestResponse(BaseModel):
    status: str
    vector_store_loaded: bool


@app.on_event("startup")
def startup_event():
    """Attempt to load existing vector store at startup."""
    global vector_store, image_data_store
    try:
        vs, ids = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)
        vector_store, image_data_store = vs, ids
        print("Vector store loaded on startup.")
    except Exception as e:
        print(f"Vector store not available at startup: {e}")
        vector_store, image_data_store = None, None


@app.get("/health")
def health():
    return {"status": "ok", "vector_store_loaded": bool(vector_store)}


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Run the ingestion process to re-create the vector store from `data/`.

    This calls the same code as `ingest.py` and reloads the vector store after completion.
    """
    global vector_store, image_data_store
    try:
        ingest_main()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    try:
        vs, ids = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)
        vector_store, image_data_store = vs, ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store not found after ingestion: {e}")

    return {"status": "ingested", "vector_store_loaded": True}


@app.post("/query")
def query(request: QueryRequest):
    """Query the RAG pipeline and return an answer and retrieved context."""
    global vector_store, image_data_store
    if not vector_store or not image_data_store:
        raise HTTPException(status_code=503, detail="Vector store is not available. Run /ingest first.")

    q = request.query
    k = request.top_k

    # Retrieve context docs
    retrieved_docs = retrieval_multimodal(q, vector_store, k=k)

    # Run LLM pipeline to get answer text
    try:
        answer = multimodal_rag_pipeline(q, vector_store, image_data_store, LLM)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM pipeline failed: {e}")

    # Build a light-weight serializable representation of retrieved docs
    retrieved = []
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        entry = {
            "type": meta.get("type", "text"),
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", None),
        }
        if entry["type"] == "text":
            entry["text"] = doc.page_content
        else:
            # image
            image_id = meta.get("image_id")
            entry["image_id"] = image_id
            if request.include_images and image_id in image_data_store:
                entry["image_base64"] = f"data:image/png;base64,{image_data_store[image_id]}"
        retrieved.append(entry)

    return {"query": q, "answer": answer, "retrieved": retrieved}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("project.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
