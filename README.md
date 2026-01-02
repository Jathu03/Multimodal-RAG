# 🧠 Multimodal RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system that allows users to chat with PDF documents containing both **text and images**.

This system uses **CLIP** for multimodal embeddings (mapping text and images into the same vector space) and a **Vision-Language Model (e.g., GPT-4o)** to generate answers grounded in retrieved multimodal context.

---

## 🚀 Features

- **Multimodal Ingestion**: Automatically extracts text and images from PDFs.
- **Unified Vector Search**: Uses CLIP to embed both text chunks and images into a single FAISS index.
- **RAG Pipeline**: Retrieves the most relevant text _and_ images to answer user queries.
- **Modern Architecture**:

  - **Backend**: FastAPI (async, type-safe APIs).
  - **Frontend**: Streamlit (interactive, user-friendly UI).
  - **Orchestration**: LangChain.

- **DevOps Ready**: Dockerized services, Docker Compose orchestration, and GitHub Actions CI/CD.

---

## 📂 Project Structure

```text
.
├── backend/                  # FastAPI application
│   ├── Dockerfile
│   └── app.py
├── frontend/                 # Streamlit UI
│   ├── Dockerfile
│   └── app.py
├── src/                      # Core logic package
│   ├── config.py             # Configuration & lazy model loading
│   ├── data_processing.py    # PDF extraction & CLIP embedding
│   ├── rag_pipeline.py       # Retrieval & prompt engineering
│   └── vector_store_utils.py # FAISS & pickle management
├── tests/                    # Unit tests (pytest)
├── vector_store/             # Persisted FAISS index (gitignored)
├── .env.example              # Environment variable template
├── .gitignore
├── .dockerignore
├── docker-compose.yml        # Service orchestration
└── requirements.txt          # Python dependencies
```

---

## 🛠️ Installation & Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
```

### 2️⃣ Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env .env
```

Edit `.env` and add your API keys and paths:

```ini
OPENAI_API_KEY=sk-proj-your-key-here
DOCUMENTS_PATH=data
VECTOR_STORE_PATH=vector_store
IMAGE_STORE_PATH=vector_store/images.pkl
```

### 3️⃣ Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Option A: Using Docker (Recommended)

Run the entire stack (backend + frontend) with Docker Compose:

```bash
docker-compose up --build
```

Once running, access the services:

- **Frontend UI**: [http://localhost:8501](http://localhost:8501)
- **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Running Manually (Development Mode)

You will need two terminal windows.

#### Terminal 1 — Backend

```bash
# Ensure virtual environment is activated
python backend/app.py
or
uvicorn app:app --reload
```

Wait until you see:

```text
Application startup complete
```

#### Terminal 2 — Frontend

```bash
# Ensure virtual environment is activated
streamlit run frontend/app.py
```

---

## 🧪 Testing

The project includes a comprehensive test suite using **pytest**. All external dependencies (GPU, CLIP, OpenAI APIs) are mocked to ensure fast and deterministic tests.

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v
```

---

## ⚙️ How It Works

### 1️⃣ Ingestion

- A PDF is uploaded through the Streamlit UI.
- The file is sent to the FastAPI backend.

### 2️⃣ Extraction

- **PyMuPDF** extracts:

  - Text blocks
  - Embedded images

### 3️⃣ Embedding (Multimodal)

- **CLIP (Contrastive Language–Image Pretraining)** converts:

  - Text chunks → vector embeddings
  - Images → vector embeddings

Because both are mapped into the **same vector space**, the system supports:

- Text → Text search
- Text → Image search
- Image → Text search (extensible)

### 4️⃣ Storage

- Vector embeddings are stored in a **FAISS index**.
- Raw base64-encoded images are stored in a **pickle file** for later retrieval.

### 5️⃣ Retrieval

- User queries are embedded using CLIP.
- FAISS retrieves the nearest vectors (text and/or images).

### 6️⃣ Generation

- Retrieved text and base64 images are formatted into a structured prompt.
- The prompt is sent to a **Vision-Language Model** (e.g., GPT-4o).
- The model generates a grounded, multimodal-aware response.

---

## 🚢 CI/CD Pipeline

GitHub Actions workflows are configured under `.github/workflows/`:

### ✅ Continuous Integration (CI — `ci.yml`)

- Triggered on every push and pull request to `main`.
- Runs:

  - Unit tests (`pytest`)
  - Linting and sanity checks (if configured)

### 📦 Continuous Deployment (CD — `cd.yml`)

- Triggered on pushes to `main`.
- Automatically:

  - Builds Docker images
  - Pushes images to **GitHub Container Registry (GHCR)**

---

## 📌 Summary

This project demonstrates a **production-grade multimodal RAG system**, combining:

- Vision–language embeddings (CLIP)
- Vector databases (FAISS)
- Modern backend/frontend stacks (FastAPI + Streamlit)
- Robust DevOps practices (Docker + GitHub Actions)

It is suitable as both a **research prototype** and a **real-world deployable system**.

