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
from .config import CLIP_MODEL, CLIP_PROCESSOR, TEXT_SPLITTER

# --- Embedding Functions  ---
def embed_image(image_data):
    """Embed image using CLIP."""
    image = image_data.convert("RGB")
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")
    with torch.no_grad():
        features = CLIP_MODEL.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text_document):
    """Embed text using CLIP."""
    inputs = CLIP_PROCESSOR(
        text=text_document.page_content, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        features = CLIP_MODEL.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# --- PDF Processing Function ---
def process_pdf(pdf_path):
    """
    Processes a single PDF, extracting and embedding text and images.
    Returns lists of documents, embeddings, and a dictionary of image data.
    """
    doc_name = os.path.basename(pdf_path)
    print(f"Processing document: {doc_name}...")
    
    docs = []
    embeddings = []
    image_data = {}
    
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text", "source": doc_name})
            text_chunks = TEXT_SPLITTER.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk)
                embeddings.append(embedding)
                docs.append(chunk)

        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                image_id = f"{doc_name}_page_{i}_img_{img_index}"
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data[image_id] = img_base64

                embedding = embed_image(pil_image)
                embeddings.append(embedding)

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id, "source": doc_name}
                )
                docs.append(image_doc)
            except Exception as e:
                print(f"Error processing image {img_index} on page {i} from {doc_name}: {e}")
    doc.close()
    return docs, embeddings, image_data