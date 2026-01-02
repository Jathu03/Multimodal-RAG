# Recommended Project Structure — Multimodal RAG 🚀

## Short recommendation

- Keep `main.py` at the **project root** (or in `scripts/`) — keep `src/` as an importable package.

---

## Project layout (suggested)

```
.
├─ README.md
├─ LICENSE
├─ .env.example
├─ requirements.txt / environment.yml / pyproject.toml
├─ Dockerfile
├─ docker-compose.yml        # optional: orchestrate services
├─ .github/
│  └─ workflows/
│     ├─ ci.yml
│     └─ cd.yml              # optional
├─ src/                      # core, importable package
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_processing.py
│  ├─ rag_pipeline.py
│  ├─ vector_store_utils.py
│  └─ description.md         # high-level summaries (optional)
├─ scripts/                  # CLI helpers (optional)
│  └─ rebuild_index.py
├─ main.py                   # CLI runner / smoke-test (recommended at repo root)
├─ backend/                  # FastAPI service
│  ├─ app.py
│  ├─ api/
│  └─ Dockerfile
├─ frontend/                 # Streamlit UI
│  ├─ app.py
│  └─ Dockerfile
├─ data/                     # PDFs and sample inputs (gitignored or small samples)
├─ vector_store/             # persisted FAISS index + image pickles (gitignored)
│  └─ faiss_index/
├─ tests/
│  ├─ test_data_processing.py
│  └─ test_rag_pipeline.py
└─ docs/
   └─ architecture.md
```

---

## Roles & integration notes 🔧

- `src/`: reusable core logic used by both backend and frontend.
- `main.py` (root): CLI script for building/indexing and quick smoke tests (used by devs and CI).
- `backend/` (FastAPI): exposes API endpoints such as `/query`, `/ingest` (optional) and loads the vector store to serve requests.
- `frontend/` (Streamlit): UI that calls the backend endpoints; displays text + images (data URLs).
- `vector_store/`: keep out of VCS (add to `.gitignore`). Persist as artifact, volume, or rebuild on deploy.
- `Docker`: create service-specific Dockerfiles for backend and frontend and orchestrate with `docker-compose` in dev.
- `CI/CD`: use GitHub Actions to run linting, unit tests, and an optional smoke test that runs `main.py --no-rebuild` against a small sample dataset.

---

## Practical tips ✅

- Add `--limit N` and `--dry-run` flags to `main.py` for quick CI smoke tests.
- Make `src/config.py` read env vars but avoid heavy model initialization at import time (or add lazy-loading helpers) to speed up tests and CI.
- Provide `.env.example` and document expected secrets (API keys) in `README.md`.
- Add unit tests for `process_pdf`, `create_and_save_vector_store`, and `multimodal_rag_pipeline` (mock external dependencies).

---

## Next steps (optional) ✨

- I can move `main.py` to the repo root (if you prefer) and add skeleton `backend/app.py` and `frontend/app.py` files.
- I can also add basic `docker-compose.yml` + per-service `Dockerfile` templates and a minimal GitHub Actions `ci.yml`.

Would you like me to apply any of these changes now?
