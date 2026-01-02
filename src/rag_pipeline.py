# --- Standard Library Imports ---
import json

# --- Third-Party Library Imports ---
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- Internal Imports ---
from src.data_processing import embed_text

def retrieval_multimodal(query, vector_store, k=5):
    """Unified retrieval using CLIP embeddings."""
    query_embedding = embed_text(Document(page_content=query))
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """
    Formats retrieved text and image data into a single multimodal message for an LLM.

    Inputs:
        query (str): The original user question.
        retrieved_docs (List[Document]): Context documents (text snippets or image IDs).
        image_data_store (dict): Mapping of image IDs to Base64-encoded strings.

    Action:
        Builds a LangChain HumanMessage with a content list containing:
        1. Text excerpts: Extracted from text-based Document objects.
        2. Embedded images: Formatted as data URLs (data:image/jpeg;base64,...) 
           using the image_data_store.
        3. The original query: Appended to provide the final instruction.

    Output:
        HumanMessage: A structured object where the 'content' attribute is a 
        list of text and image_url dictionaries.
    """
    content = [{"type": "text", "text": f"Question: {query}\n\nContent:\n"}]
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # 1. Process retrieved context
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

    # 2. Add the user's query
    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return HumanMessage(content=content)

def multimodal_rag_pipeline(query, vector_store, image_data_store, llm):
    """
    Executes the full Multimodal RAG pipeline: Retrieval -> Formatting -> Generation.

    Inputs:
        query (str): The user's input question.
        vector_store (FAISS): The pre-loaded vector database.
        image_data_store (dict): The dictionary containing Base64 image data.
        llm (BaseChatModel): A vision-capable LangChain LLM instance.

    Action:
        1. Retrieves relevant context documents via retrieval_multimodal.
        2. Builds a multimodal message using create_multimodal_message.
        3. Prints the retrieved source metadata for transparency/debugging.
        4. Invokes the LLM with the constructed message: llm.invoke([message]).

    Output:
        str: The content attribute (response text) from the LLM call.
    """
    # 1. Retrieval
    context_docs = retrieval_multimodal(query, vector_store, k=5)

    # 2. Formatting
    message = create_multimodal_message(query, context_docs, image_data_store)
    
    # 3. Logging/Citations
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
    
    # 4. Generation
    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    print("Running quick self-check for rag_pipeline...")
    from langchain_core.documents import Document

    docs = [Document(page_content="This is a test.", metadata={"page":0,"type":"text","source":"test.pdf"})]
    image_data = {}

    msg = create_multimodal_message("What is this?", docs, image_data)
    print("Created HumanMessage content items:", len(getattr(msg, "content", [])))
    if getattr(msg, "content", []):
        print("First content item:", msg.content[0])