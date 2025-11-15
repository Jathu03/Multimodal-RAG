# --- Standard Library Imports ---
import json

# --- Third-Party Library Imports ---
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- Internal Imports ---
from .data_processing import embed_text

def retrieval_multimodal(query, vector_store, k=5):
    """Unified retrieval using CLIP embeddings."""
    query_embedding = embed_text(Document(page_content=query))
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Create a message with both text and images for the LLM."""
    content = [{"type": "text", "text": f"Question: {query}\n\nContent:\n"}]
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join([f"[Source: {doc.metadata['source']}, Page {doc.metadata['page']}]: {doc.page_content}" for doc in text_docs])
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from {doc.metadata['source']}, page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })

    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return HumanMessage(content=content)

def multimodal_rag_pipeline(query, vector_store, image_data_store, llm):
    """Main pipeline for multimodal RAG."""
    context_docs = retrieval_multimodal(query, vector_store, k=5)
    message = create_multimodal_message(query, context_docs, image_data_store)
    
    print("\n--- Retrieved Context ---")
    for doc in context_docs:
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "?")
        doc_type = doc.metadata.get("type", "unknown")
        if doc_type == "text":
            print(f"  - [Text] Source: {source}, Page: {page}")
        else:
            print(f"  - [Image] Source: {source}, Page: {page}")
    print("-------------------------\n")
    
    response = llm.invoke([message])
    return response.content