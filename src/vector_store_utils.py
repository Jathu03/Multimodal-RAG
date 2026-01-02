# --- Standard Library Imports ---
import pickle
import numpy as np

# --- Third-Party Library Imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def create_and_save_vector_store(docs, embeddings, image_data, vector_store_path, image_store_path):
    """
    Creates a FAISS vector store from pre-computed embeddings and persists it to disk.

    Inputs:
        docs (List[Document]): List of LangChain Documents (text or image placeholders).
        embeddings (List[np.ndarray]): List of normalized CLIP feature vectors.
        image_data (Dict[str, str]): Dictionary mapping image IDs to Base64 strings.
        vector_store_path (str): Directory path for the FAISS index.
        image_store_path (str): File path for the image metadata pickle.

    Actions:
        1. Validation: If the embeddings list is empty, prints a warning and exits.
        2. Conversion: Transforms the embeddings list into a unified NumPy array.
        3. Indexing: Calls FAISS.from_embeddings() to pair document text with 
           their corresponding vectors.
        4. FAISS Persistence: Saves the resulting vector store to disk.
        5. Image Persistence: Pickles the image_data dictionary to the specified path.

    Output:
        None (persists data to specified file paths).
    """

    # 1. Validation: Ensure we have data to process
    if not embeddings:
        print("No embeddings generated. Skipping vector store creation.")
        return

    # 2. Conversion: Standardize embeddings for FAISS compatibility
    print("Creating FAISS vector store...")
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # 3. Indexing: Map text content to vectors in the shared latent space
    # Pair each document's content with its respective embedding and metadata
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings_array)],
        embedding=None, # Provide the dummy class instance here
        metadatas=[doc.metadata for doc in docs]
    )
    
    # 4. FAISS Persistence: Save the searchable index locally
    print(f"Saving vector store to {vector_store_path}")
    vector_store.save_local(vector_store_path)
    
    # 5. Image Persistence: Save the Base64 image lookup table
    print(f"Saving image data to {image_store_path}")
    with open(image_store_path, "wb") as f:
        pickle.dump(image_data, f)

def load_vector_store(vector_store_path, image_store_path):
    """
    Loads a persisted FAISS index and image data dictionary from disk.

    Inputs:
        vector_store_path (str): The directory path where the FAISS index is stored.
        image_store_path (str): The file path of the pickled image data.

    Actions:
        1. Loads the FAISS vector store using FAISS.load_local(). 
           Note: allow_dangerous_deserialization is set to True to enable pickle loading.
        2. Deserializes and loads the pickled image_data dictionary from the image_store_path.

    Output:
        tuple: (vector_store, image_data_store)
    """

    # 1. Load the FAISS index
    # We pass embeddings=None because the index already contains the vectors
    print("Loading vector store and image data...")
    vector_store = FAISS.load_local(
        vector_store_path, 
        embeddings=None, 
        allow_dangerous_deserialization=True
    )
    # 2. Load the image lookup table
    with open(image_store_path, "rb") as f:
        image_data_store = pickle.load(f)
    
    return vector_store, image_data_store


def update_vector_store(new_docs, new_embeddings, new_image_data, vector_store_path, image_store_path):
    """
    Loads an existing vector store, adds new documents/embeddings, and saves it.
    """
    import os
    
    # 1. Check if store exists. If not, create new.
    if not os.path.exists(vector_store_path) or not os.path.exists(image_store_path):
        print("No existing store found. Creating new one.")
        create_and_save_vector_store(new_docs, new_embeddings, new_image_data, vector_store_path, image_store_path)
        return

    # 2. Load existing resources
    print("Loading existing vector store to update...")
    vector_store, existing_image_data = load_vector_store(vector_store_path, image_store_path)

    # 3. Add new embeddings to FAISS
    # Convert list to standard format matching the 'create' function logic
    if new_embeddings:
        print(f"Adding {len(new_embeddings)} new embeddings to store...")
        embeddings_array = np.array(new_embeddings, dtype=np.float32)
        
        # Zip content and embeddings together
        text_embeddings_pairs = [
            (doc.page_content, emb) 
            for doc, emb in zip(new_docs, embeddings_array)
        ]
        
        # Add to FAISS index
        vector_store.add_embeddings(
            text_embeddings=text_embeddings_pairs,
            metadatas=[doc.metadata for doc in new_docs]
        )

    # 4. Merge Image Data
    existing_image_data.update(new_image_data)

    # 5. Save everything back to disk
    print("Persisting updated store...")
    vector_store.save_local(vector_store_path)
    with open(image_store_path, "wb") as f:
        pickle.dump(existing_image_data, f)
    
    print("Update complete.")