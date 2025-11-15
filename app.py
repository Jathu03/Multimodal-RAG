# --- Internal Imports ---
from src.config import VECTOR_STORE_PATH, IMAGE_STORE_PATH, LLM
from src.vector_store_utils import load_vector_store
from src.rag_pipeline import multimodal_rag_pipeline

def main():
    """
    Loads the vector store and starts an interactive query session.
    """
    try:
        vector_store, image_data_store = load_vector_store(VECTOR_STORE_PATH, IMAGE_STORE_PATH)
    except FileNotFoundError:
        print("Vector store not found. Please run 'python ingest.py' first to create it.")
        return
    
    print("Multimodal RAG system is ready. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        print("\nThinking...")
        answer = multimodal_rag_pipeline(query, vector_store, image_data_store, LLM)
        print(f"\nAnswer: {answer}")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()