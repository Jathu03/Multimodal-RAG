import os
import shutil
import logging
from typing import Optional
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# Add the PARENT directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- Internal Imports ---
# These assume the project root is in PYTHONPATH
from src.config import DOCUMENTS_PATH, VECTOR_STORE_PATH, IMAGE_STORE_PATH, LLM
from src.data_processing import process_pdf
from src.vector_store_utils import update_vector_store, load_vector_store
from src.rag_pipeline import multimodal_rag_pipeline

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI(title="Multimodal RAG API", version="1.0")

# Global variables to hold the loaded index in memory
vector_store = None
image_data_store = {}

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: Optional[list] = None

# --- Lifespan Events (Startup) ---
@app.on_event("startup")
async def startup_event():
    """
    Load the Vector Store and Image Data once when the server starts.
    If they don't exist yet, we initialize empty variables and wait for uploads.
    """
    global vector_store, image_data_store
    
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(IMAGE_STORE_PATH):
        logger.info("Loading existing vector store...")
        try:
            vector_store, image_data_store = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
    else:
        logger.warning("No vector store found. Please upload documents via the /upload endpoint.")

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check to ensure API is running."""
    return {"status": "ok", "vector_store_loaded": vector_store is not None}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Main endpoint: Accepts a text query, runs Multimodal RAG, and returns the answer.
    """
    global vector_store, image_data_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not loaded. Upload a document first.")

    try:
        logger.info(f"Received query: {request.query}")
        
        # Run the full pipeline
        # Note: In a real prod app, you might want to run this in a threadpool to not block async loop
        response_text = multimodal_rag_pipeline(
            request.query, 
            vector_store, 
            image_data_store, 
            LLM
        )
        
        return QueryResponse(answer=response_text)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, process it, and update the in-memory and on-disk Vector Store.
    """
    global vector_store, image_data_store

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # 1. Save file to disk
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        file_path = os.path.join(DOCUMENTS_PATH, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing uploaded file: {file.filename}")

        # 2. Process PDF (Extract Text & Images)
        docs, embeddings, image_data = process_pdf(file_path)

        if not embeddings:
            return {"message": "File uploaded, but no content found to index."}

        # 3. Update Vector Store (Disk)
        update_vector_store(
            docs, 
            embeddings, 
            image_data, 
            VECTOR_STORE_PATH, 
            IMAGE_STORE_PATH
        )

        # 4. Reload Vector Store (Memory)
        # We reload to ensure the running API has the latest data
        vector_store, image_data_store = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)

        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_added": len(docs),
            "images_added": len(image_data)
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run server locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)