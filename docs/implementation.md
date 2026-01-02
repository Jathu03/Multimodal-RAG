# Implementation Plan: Multimodal RAG System

This document outlines the step-by-step execution plan to build, test, and deploy the Multimodal RAG system. Follow this order to ensure a stable environment, reproducible builds, and a logical progression from core logic to user interface.

---

## 🏗 Phase 1: Environment & Foundations

### 1. Initialize Repo & Environment 🧰

**Goal:** Create a reproducible, safe development environment.

- **Git Setup:** Initialize repository.
- **.gitignore:** Create file and exclude:
  - `.venv/`
  - `vector_store/`
  - `*.pkl`
  - `.env`
  - `__pycache__/`
- **Environment Variables:** Create `.env.example` (template) and `.env` (actual secrets).
- **Virtual Env:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  ```
- **Dependencies:** Create `requirements.txt`. Install initial libs (pip install ...) and freeze.

### 2. Create Source Package Skeleton 📦

**Goal:** Keep code modular and importable by backend, frontend, and tests.

- **Directory Structure:** Create a `src/` folder.
- **Init:** Add `src/__init__.py` to make it a package.

### 3. Configuration & Lazy Loading 🔧

**Goal:** Centralize configuration and prevent slow startup times.

- **Create `src/config.py`:**
  - Load environment variables (API keys).
  - Define path constants: `DOCUMENTS_PATH`, `VECTOR_STORE_PATH`, `IMAGE_STORE_PATH`.
  - **Crucial:** Implement lazy model initialization helpers (e.g., `get_clip_model()`, `get_llm()`) so heavy models are only loaded when actually needed, not at import time.

---

## 🧩 Phase 2: Core Logic Implementation

### 4. Implement Core Modules (Order Matters)

**Goal:** Build the heart of ingestion and retrieval before worrying about the UI.

- **`src/data_processing.py`**
  - Implement `process_pdf`, `embed_image`, `embed_text`.
  - **Tip:** Add `--limit` or `--dry-run` logic to functions to allow fast testing without processing huge datasets.
- **`src/vector_store_utils.py`**
  - Implement `create_and_save_vector_store` and `load_vector_store`.
  - Ensure strictly defined FAISS API usage.
  - Create a small `Embeddings` wrapper/mock for testing purposes.
- **`src/rag_pipeline.py`**
  - Implement `retrieval_multimodal`.
  - Implement `create_multimodal_message` (prompt engineering).
  - Implement `multimodal_rag_pipeline`.

### 5. CLI & Smoke Test Entry Point ✅

**Goal:** Manual verification and CI smoke testing.

- **Create `main.py` (Root Level):**
  - Glue logic: Ingestion → Build Index → Load Index → Query.
  - **Add Flags:**
    - `--no-rebuild`: Skip expensive PDF processing/embedding.
    - `--limit [n]`: Only process _n_ pages/images.
    - `--query "..."`: Run a specific test query and exit.

### 6. Unit Tests & Test Data ⚠️

**Goal:** Prevent regressions and enable fast CI.

- **Test Setup:**
  - Create `tests/` directory.
  - Add small sample PDFs/Images in `tests/data/`.
- **Implement Tests:**
  - `tests/test_data_processing.py`
  - `tests/test_vector_store_utils.py`
  - `tests/test_rag_pipeline.py`
- **Strategy:** Mock heavy dependencies (CLIP, LLM APIs) to ensure tests run in seconds, not minutes.

---

## 🌐 Phase 3: Application Layer

### 7. Backend Skeleton (FastAPI)

**Goal:** Separate serving concerns from core logic.

- **Setup:** Create `backend/` directory.
- **`backend/app.py`:**
  - Initialize `FastAPI` app.
  - Load Vector Store **once** at startup (lifespan event).
  - Implement `POST /query` endpoint.
  - Implement `GET /health` endpoint.
  - Add CORS middleware.

### 8. Frontend Skeleton (Streamlit) 🎛️

**Goal:** Rapid, developer-friendly UI.

- **Setup:** Create `frontend/` directory.
- **`frontend/app.py`:**
  - Create input field for user query.
  - Call backend `POST /query`.
  - Render text response.
  - Render base64 images neatly.

---

## 🚀 Phase 4: Deployment & DevOps

### 9. Dockerize Services 🐳

**Goal:** Reproducible deployments and easy local orchestration.

- **`backend/Dockerfile`:** Python base, install requirements, run uvicorn.
- **`frontend/Dockerfile`:** Python base, install requirements, run streamlit.
- **`docker-compose.yml`:**
  - Define services for backend and frontend.
  - **Volume Strategy:** Decide how to persist `vector_store` (mount existing local folder or build inside container).

### 10. CI/CD (GitHub Actions) 🔁

**Goal:** Automation and safety.

- **`.github/workflows/ci.yml`:**
  - Linting (flake8/black).
  - Unit Tests (pytest).
  - Smoke Test: Run `python main.py --no-rebuild --limit 1 --query "test"`.
- **`.github/workflows/cd.yml` (Optional):**
  - Build & Push Docker images on merge to main.

### 11. Documentation & Housekeeping 📚

**Goal:** Onboarding and clarity.

- **`README.md`:** Installation, usage, and "How to run smoke tests".
- **`docs/architecture.md`:** Diagram of how the pieces fit together.

---

## ⚠️ Practical Tips & Safety Notes

1.  **Lazy Initialization is King:** Use `get_clip_model()` factory functions. If you load CLIP at the top level of `src/data_processing.py`, your simple unit tests will take 30 seconds to start.
2.  **Git Large File Storage:** Keep the actual `vector_store/` folder **out of git**.
3.  **CI Speed:** Provide extremely small sample PDFs for CI. Do not try to process real books in GitHub Actions.
4.  **Mocking:** In `tests/`, mock the OpenAI/LLM calls. You don't want to pay for API credits every time you run `pytest`.

---

## ✅ Quick Checklist (Start Here)

- `git init`, add `.gitignore`, create `.env.example`
- Create `src/` folder and `src/config.py` (with lazy init logic)
- Implement `src/data_processing.py` and `src/vector_store_utils.py`
- Implement `src/rag_pipeline.py` and root `main.py`
- Add tests and ensure `python main.py --no-rebuild --limit 1` works
- Scaffold `backend/` and `frontend/`
- Create Dockerfiles and `docker-compose.yml`
- Add GitHub Actions workflows
