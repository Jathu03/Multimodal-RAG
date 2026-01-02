import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.documents import Document
from src.vector_store_utils import create_and_save_vector_store, update_vector_store, load_vector_store

@pytest.fixture
def sample_data():
    docs = [Document(page_content="test", metadata={"source": "test"})]
    embeddings = [np.random.rand(512).astype(np.float32)]
    image_data = {"img1": "base64string"}
    return docs, embeddings, image_data

@patch('src.vector_store_utils.FAISS')
@patch('src.vector_store_utils.pickle')
def test_create_and_save(mock_pickle, mock_faiss, sample_data):
    """Test that FAISS creation and saving is called correctly."""
    docs, embeddings, image_data = sample_data
    
    # Run
    create_and_save_vector_store(docs, embeddings, image_data, "v_path", "i_path")
    
    # Check FAISS from_embeddings was called
    mock_faiss.from_embeddings.assert_called_once()
    
    # Check if save_local was called on the instance
    mock_vector_store = mock_faiss.from_embeddings.return_value
    mock_vector_store.save_local.assert_called_with("v_path")
    
    # Check if pickle dump was called for images
    # We mock 'open' inside the function implicitly via the with statement in logic? 
    # Actually, we need to patch built-in open for pickle
    with patch("builtins.open", mock_open()) as mock_file:
        create_and_save_vector_store(docs, embeddings, image_data, "v_path", "i_path")
        mock_pickle.dump.assert_called()

@patch('src.vector_store_utils.load_vector_store')
@patch('src.vector_store_utils.create_and_save_vector_store')
@patch('os.path.exists')
def test_update_store_creates_new_if_missing(mock_exists, mock_create, mock_load, sample_data):
    """If store doesn't exist, update should call create instead."""
    docs, embeddings, image_data = sample_data
    mock_exists.return_value = False  # Simulate file missing
    
    update_vector_store(docs, embeddings, image_data, "v_path", "i_path")
    
    mock_create.assert_called_once()
    mock_load.assert_not_called()

@patch('src.vector_store_utils.load_vector_store')
@patch('os.path.exists')
def test_update_store_appends_if_exists(mock_exists, mock_load, sample_data):
    """If store exists, it should load and add_embeddings."""
    docs, embeddings, image_data = sample_data
    mock_exists.return_value = True
    
    # Mock the loaded store
    mock_faiss_instance = MagicMock()
    mock_load.return_value = (mock_faiss_instance, {}) 
    
    with patch("builtins.open", mock_open()):
        with patch("src.vector_store_utils.pickle.dump"):
            update_vector_store(docs, embeddings, image_data, "v_path", "i_path")
            
            # Verify add_embeddings was called
            mock_faiss_instance.add_embeddings.assert_called()
            mock_faiss_instance.save_local.assert_called()