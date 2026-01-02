# Project Overview — Multimodal RAG

## Summary of files ✅

### `src/config.py` 🔧

- Purpose: Central configuration and global initializations.
- What it does:
  - Loads environment variables and sets `GOOGLE_API_KEY`.
  - Initializes models and utilities: **`CLIP_MODEL`**, **`CLIP_PROCESSOR`**, **`LLM`** (`ChatGoogleGenerativeAI`), and **`TEXT_SPLITTER`**.
  - Defines path constants: `BASE_DIR`, `DOCUMENTS_PATH`, `VECTOR_STORE_PATH`, `IMAGE_STORE_PATH`.

### `src/data_processing.py` 📄

- Purpose: Parse PDFs, extract text and images, and compute CLIP embeddings.
- Functions:
  - `embed_image(image_data)` — takes a PIL Image, returns a normalized CLIP image embedding (numpy array).
  - `embed_text(text_document)` — takes a `Document` with `page_content`, returns a normalized CLIP text embedding (numpy array).
  - `process_pdf(pdf_path)` — opens a PDF, extracts text and images per page, splits text into chunks using `TEXT_SPLITTER`, embeds each chunk/image, and returns `(docs, embeddings, image_data)` where:
    - `docs` is a list of `Document` objects (metadata includes `page`, `type`, `source`, `image_id`),
    - `embeddings` is a list of numpy vectors,
    - `image_data` maps image IDs to base64 PNG strings.

### `src/vector_store_utils.py` 🗄️

- Purpose: Build, persist, and load a FAISS vector store and image data.
- Functions:
  - `create_and_save_vector_store(docs, embeddings, image_data, vector_store_path, image_store_path)` — constructs FAISS index from `docs` and `embeddings`, saves index and pickled `image_data`.
  - `load_vector_store(vector_store_path, image_store_path)` — loads FAISS index and returns `(vector_store, image_data_store)`.

### `src/rag_pipeline.py` 🧠

- Purpose: Multimodal retrieval and LLM prompt construction (text + images).
- Functions:
  - `retrieval_multimodal(query, vector_store, k=5)` — embeds query via `embed_text` and runs `vector_store.similarity_search_by_vector`.
  - `create_multimodal_message(query, retrieved_docs, image_data_store)` — builds a `HumanMessage` containing text excerpts and inline images (data URLs using base64 data from `image_data_store`).
  - `multimodal_rag_pipeline(query, vector_store, image_data_store, llm)` — retrieves context, builds message, prints sources, invokes `llm.invoke([message])`, and returns `response.content`.

---

## Dataflow 🔁

1. `config.py` initializes models (`CLIP`, processor, `TEXT_SPLITTER`, `LLM`) and path constants.
2. `data_processing.process_pdf(pdf_path)` reads PDFs, extracts text & images, splits text, calls `embed_text` / `embed_image` to produce `docs`, `embeddings`, and `image_data`.
3. `vector_store_utils.create_and_save_vector_store(...)` builds and saves FAISS index and pickled `image_data`.
4. At runtime, `vector_store_utils.load_vector_store(...)` loads `vector_store` and `image_data_store`.
5. `rag_pipeline.multimodal_rag_pipeline(query, vector_store, image_data_store, llm)` performs retrieval and constructs a multimodal message to `llm`.

Visual: `config.py` → `data_processing.process_pdf` → `create_and_save_vector_store` → (saved index & images) → `load_vector_store` → `rag_pipeline.multimodal_rag_pipeline`

---

## Notes & Potential Issues ⚠️

- `vector_store_utils.create_and_save_vector_store` references `DummyEmbeddings()` which is undefined — will raise a `NameError`.
- The `FAISS.from_embeddings` call signature may not match the installed FAISS wrapper — verify with your package version.
- `create_multimodal_message` uses `HumanMessage(content=...)` with a list of dicts — confirm the `HumanMessage` API supports this structure.
- `config.py` sets `os.environ["GOOGLE_API_KEY"] = os.getenv("Gemini_key")` — ensure your `.env` variable matches.

---

## Next steps (optional) ✨

- Fix undefined `DummyEmbeddings` and ensure FAISS creation call matches your library.
- Add a small README or tests demonstrating ingestion → index → retrieval flow.
- Add lightweight unit tests or a sample driver script.

If you want, I can implement any of these fixes or add the tests next.
