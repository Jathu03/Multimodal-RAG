# --- Standard Library Imports ---
import base64
import io
import os

# --- Third-Party Library Imports ---
import torch
import fitz  # PyMuPDF
from PIL import Image
from langchain_core.documents import Document

# --- Internal Imports ---
from src.config import CLIP_MODEL, CLIP_PROCESSOR, TEXT_SPLITTER

# --- Embedding Functions  ---
def embed_image(image_data):
    """
    Embed image using CLIP.

    Input: 
        image_data: A PIL Image object.
    Action: 
        Runs CLIP image encoder to produce a normalized embedding (numpy vector).
    Output: 
        NumPy embedding vector.
    """

    # Ensure image is in RGB mode for CLIP consistency
    image = image_data.convert("RGB")

    # Preprocess and move tensors to appropriate device
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")
    with torch.no_grad():
        features = CLIP_MODEL.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text_document):
    """
    Embed text using CLIP.

    Input: 
        text_document: Document (langchain) with page_content.
    Action: 
        Uses CLIP text encoder to produce a normalized embedding (numpy vector).
    Output: 
        NumPy embedding vector.
    """

    # Tokenize and encode the text chunk
    inputs = CLIP_PROCESSOR(
        text=text_document.page_content, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        # Extract features and L2-normalize
        features = CLIP_MODEL.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# --- PDF Processing Function ---
def process_pdf(pdf_path):
    """
    Processes a single PDF, extracting and embedding text and images.

    Input: 
        pdf_path: Path to a PDF file.
    Actions:
        1. PDF Opening: Opens file with PyMuPDF (fitz) and iterates through pages.
        2. Text Extraction: Extracts page text, wraps into Documents, splits them 
           via TEXT_SPLITTER, and embeds each chunk.
        3. Image Extraction: Identifies images, converts to PNG base64 strings 
           for image_data dict, embeds them, and creates Document placeholders.
    Output: 
        tuple (docs, embeddings, image_data):
        - docs: List of Document objects with rich metadata.
        - embeddings: List of numpy vectors.
        - image_data: Dict mapping image IDs to base64 PNG data.
    """
    doc_name = os.path.basename(pdf_path)
    print(f"Processing document: {doc_name}...")
    
    docs = []
    embeddings = []
    image_data = {}
    
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        # --- Stage 1: Process Text ---
        text = page.get_text()
        if text.strip():
            # Create a temporary document for splitting
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text", "source": doc_name})
            
            # Split into manageable chunks (fitting CLIP's 77-token limit)
            text_chunks = TEXT_SPLITTER.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk)
                embeddings.append(embedding)
                docs.append(chunk)

        # --- Stage 2: Process Images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                # Extract raw image bytes from PDF
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Generate a unique ID to link vector search with base64 storage
                image_id = f"{doc_name}_page_{i}_img_{img_index}"
                
                # Convert to Base64 for future LLM consumption (Multimodal prompt)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data[image_id] = img_base64

                # Embed the image into the same latent space as the text
                embedding = embed_image(pil_image)
                embeddings.append(embedding)

                # Create a placeholder document for the image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id, "source": doc_name}
                )
                docs.append(image_doc)
            except Exception as e:
                print(f"Error processing image {img_index} on page {i} from {doc_name}: {e}")
    doc.close()
    return docs, embeddings, image_data


if __name__ == "__main__":
    import argparse

    pdf_path = "data\multimodal_sample.pdf"

    docs, embeddings, image_data = process_pdf(pdf_path)
    print(f"Processed {len(docs)} documents, {len(embeddings)} embeddings, {len(image_data)} images")
    if docs:
        print("Sample doc metadata:", docs[0].metadata)