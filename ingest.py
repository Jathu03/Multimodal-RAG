# --- Standard Library Imports ---
import os

# --- Internal Imports ---
from src.config import DOCUMENTS_PATH, VECTOR_STORE_PATH, IMAGE_STORE_PATH
from src.data_processing import process_pdf
from src.vector_store_utils import create_and_save_vector_store

def main():
    """
    Main function to ingest all PDFs from the data directory and create a vector store.
    """
    all_docs = []
    all_embeddings = []
    all_image_data = {}

    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCUMENTS_PATH, pdf_file)
        docs, embeddings, image_data = process_pdf(pdf_path)
        all_docs.extend(docs)
        all_embeddings.extend(embeddings)
        all_image_data.update(image_data)

    create_and_save_vector_store(all_docs, all_embeddings, all_image_data, VECTOR_STORE_PATH, IMAGE_STORE_PATH)
    
    print("\nIngestion process complete.")
    print(f"Total documents processed: {len(all_docs)}")

if __name__ == "__main__":
    main()