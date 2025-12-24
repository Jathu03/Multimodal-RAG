Docker & API

This repository now provides a FastAPI server for querying the RAG pipeline and a Docker-based deployment.

- Start the API locally (after installing requirements):

```bash
pip install -r requirements.txt
uvicorn project.api:app --host 0.0.0.0 --port 8000
```

- To run in Docker:

```bash
# Build
docker build -t multimodal-rag:latest .

# Run
docker run -p 8000:8000 -e Gemin_key="$GEMINI_KEY" multimodal-rag:latest
```

- Or with docker-compose (recommended for local dev):

```bash
export GEMINI_KEY="your-api-key"
docker-compose up --build
```

Endpoints:
- `GET /health` — basic health check
- `POST /ingest` — run ingestion to (re)create the vector store from `data/`
- `POST /query` — JSON body {"query":"your question", "top_k": 5, "include_images": false}

Make sure you have your `Gemini_key` set in an environment variable or `.env` file (this project uses `python-dotenv`).
