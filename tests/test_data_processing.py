import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# --- Mock imports BEFORE importing src modules ---
# This prevents loading the heavy models
sys.modules['src.config'] = MagicMock()

from src.data_processing import embed_text, process_pdf

@patch('src.data_processing.CLIP_MODEL')
@patch('src.data_processing.CLIP_PROCESSOR')
def test_embed_text(mock_processor, mock_model):
    """Test that text embedding returns a numpy array of correct shape."""
    # Setup mock return values
    mock_processor.return_value = {"input_ids": "tensor"}
    
    # Mock the model output (features)
    mock_features = MagicMock()
    
    # --- FIX 1: Handle the in-place division ---
    # When the code does "features /= ...", return the same mock object
    # so that the subsequent .squeeze().numpy() call works on the right object.
    mock_features.__itruediv__.return_value = mock_features
    
    # Simulate a 512-dim vector result
    mock_features.norm.return_value = 1.0 
    mock_features.squeeze.return_value.numpy.return_value = np.zeros(512, dtype=np.float32)
    
    mock_model.get_text_features.return_value = mock_features

    # Create dummy doc
    doc = Document(page_content="Hello world")
    
    # Run
    result = embed_text(doc)
    
    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (512,)

@patch('src.data_processing.TEXT_SPLITTER') # --- FIX 2: Patch the Text Splitter ---
@patch('src.data_processing.fitz.open')     # Mock PyMuPDF
@patch('src.data_processing.embed_text')    # Mock the embedding call
@patch('src.data_processing.embed_image')   # Mock image embedding
def test_process_pdf(mock_embed_img, mock_embed_txt, mock_fitz_open, mock_splitter):
    """Test PDF extraction logic without reading a real file."""
    
    # 1. Setup Mock PDF structure
    mock_doc = MagicMock()
    mock_page = MagicMock()
    
    # Simulate 1 page with text
    mock_page.get_text.return_value = "This is sample text."
    # Simulate 0 images for simplicity in this test
    mock_page.get_images.return_value = []
    
    # Make the doc iterable (like: for page in doc:)
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc

    # --- FIX 2 Implementation: Configure Splitter ---
    # The splitter must return a LIST of documents, otherwise the loop in the code won't run.
    mock_splitter.split_documents.return_value = [
        Document(page_content="chunk 1", metadata={"page": 0})
    ]

    # Mock embeddings to return dummy vectors
    mock_embed_txt.return_value = np.zeros(512, dtype=np.float32)

    # 2. Run Function
    docs, embeddings, image_data = process_pdf("dummy.pdf")

    # 3. Assertions
    assert len(docs) > 0, "Should extract text documents"
    assert docs[0].page_content == "chunk 1"
    assert len(embeddings) == len(docs)
    assert isinstance(image_data, dict)