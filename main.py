"""
main.py — End-to-end pipeline for Multimodal RAG

Usage:
  python src/main.py [--rebuild] [--query "your question"]

Default behavior:
  - Scans PDFs in `DOCUMENTS_PATH` (from `src.config`).
  - Creates a `vector_store` folder (using `VECTOR_STORE_PATH` and `IMAGE_STORE_PATH`) and saves FAISS index + images.
  - Loads the vector store and prompts you for a query to run the multimodal RAG pipeline.

Notes:
  - Running this script will initialize models imported via `src.config` (may be slow).
  - Prefer to run from the project root (so `src` is importable):
    python src/main.py
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple

from src.config import DOCUMENTS_PATH, VECTOR_STORE_PATH, IMAGE_STORE_PATH
from src.data_processing import process_pdf
from src.vector_store_utils import create_and_save_vector_store, load_vector_store
from src.rag_pipeline import multimodal_rag_pipeline
from src.config import LLM


def build_vector_store_from_pdfs(pdf_paths: List[str]) -> Tuple[List, List, dict]:
    """Process PDFs and return combined (docs, embeddings, image_data)."""
    all_docs = []
    all_embeddings = []
    all_image_data = {}

    for pdf in pdf_paths:
        try:
            print(f"Processing: {pdf}")
            docs, embeddings, image_data = process_pdf(pdf)
            all_docs.extend(docs)
            all_embeddings.extend(embeddings)
            # If image ids collide between files, we prefer the first encountered
            for k, v in image_data.items():
                if k not in all_image_data:
                    all_image_data[k] = v
        except Exception as e:
            print(f"Error processing {pdf}: {e}")

    return all_docs, all_embeddings, all_image_data


def ensure_vector_store_dir_exists(vector_store_path: str, image_store_path: str):
    # Ensure parent directories exist for FAISS index and image store
    vs_parent = os.path.dirname(vector_store_path)
    img_parent = os.path.dirname(image_store_path)
    os.makedirs(vs_parent, exist_ok=True)
    os.makedirs(img_parent, exist_ok=True)


def find_pdfs(documents_path: str) -> List[str]:
    if not os.path.isdir(documents_path):
        print(f"Documents path not found: {documents_path}")
        return []
    pattern = os.path.join(documents_path, "**", "*.pdf")
    return glob.glob(pattern, recursive=True)


def interactive_query_loop(vector_store, image_data_store):
    print("Enter your query (type 'exit' or 'quit' to stop):")
    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  # newline
            break
        if not query or query.lower() in {"exit", "quit"}:
            break
        try:
            response = multimodal_rag_pipeline(query, vector_store, image_data_store, LLM)
            print("\n--- LLM Response ---")
            print(response)
            print("--------------------\n")
        except Exception as e:
            print(f"Error during pipeline: {e}")


def main(rebuild: bool = True, query: str = None):
    print("Starting Multimodal RAG pipeline...")

    # 1. Find PDFs
    pdfs = find_pdfs(DOCUMENTS_PATH)
    print(f"Found {len(pdfs)} PDF(s) under {DOCUMENTS_PATH}")

    ensure_vector_store_dir_exists(VECTOR_STORE_PATH, IMAGE_STORE_PATH)

    # 2. Optionally (re)build the vector store
    if rebuild:
        docs, embeddings, image_data = build_vector_store_from_pdfs(pdfs)
        if not embeddings:
            print("No embeddings were generated — aborting build.")
        else:
            create_and_save_vector_store(docs, embeddings, image_data, VECTOR_STORE_PATH, IMAGE_STORE_PATH)
            print("Vector store build completed.")

    # 3. Load vector store + image data
    try:
        vector_store, image_data_store = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)
    except Exception as e:
        print(f"Failed to load vector store or image data: {e}")
        sys.exit(1)

    print("Vector store loaded and ready.")

    # 4. If a single query was provided, run it once; otherwise enter REPL loop
    if query:
        try:
            response = multimodal_rag_pipeline(query, vector_store, image_data_store, LLM)
            print("\n--- LLM Response ---")
            print(response)
            print("--------------------\n")
        except Exception as e:
            print(f"Error during pipeline: {e}")
    else:
        interactive_query_loop(vector_store, image_data_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Multimodal RAG pipeline")
    parser.add_argument("--no-rebuild", dest="rebuild", action="store_false", help="Skip rebuilding the vector store")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    args = parser.parse_args()

    main(rebuild=args.rebuild, query=args.query)
