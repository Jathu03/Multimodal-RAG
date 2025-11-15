# --- Standard Library Imports ---
import pickle
import numpy as np

# --- Third-Party Library Imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def create_and_save_vector_store(docs, embeddings, image_data, vector_store_path, image_store_path):
    """Creates and saves the FAISS vector store and image data."""
    if not embeddings:
        print("No embeddings generated. Skipping vector store creation.")
        return

    print("Creating FAISS vector store...")
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings_array)],
        embedding=DummyEmbeddings(), # Provide the dummy class instance here
        metadatas=[doc.metadata for doc in docs]
    )
    
    print(f"Saving vector store to {vector_store_path}")
    vector_store.save_local(vector_store_path)
    
    print(f"Saving image data to {image_store_path}")
    with open(image_store_path, "wb") as f:
        pickle.dump(image_data, f)

def load_vector_store(vector_store_path, image_store_path):
    """Loads the FAISS vector store and image data."""
    print("Loading vector store and image data...")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embeddings=None, 
        allow_dangerous_deserialization=True
    )
    with open(image_store_path, "rb") as f:
        image_data_store = pickle.load(f)
    
    return vector_store, image_data_store