import sys
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Mock dependencies before import
sys.modules['src.config'] = MagicMock()

from src.rag_pipeline import create_multimodal_message, multimodal_rag_pipeline

def test_create_multimodal_message_logic():
    """Test formatting of text vs image context."""
    query = "What is X?"
    
    # 1. Setup Data
    text_doc = Document(page_content="X is Y", metadata={"type": "text", "source": "doc1", "page": 1})
    img_doc = Document(page_content="[Image]", metadata={"type": "image", "image_id": "img1", "source": "doc1", "page": 1})
    
    image_data_store = {"img1": "base64_fake_data"}
    
    # 2. Run
    message = create_multimodal_message(query, [text_doc, img_doc], image_data_store)
    
    # 3. Assertions
    content = message.content
    assert isinstance(content, list)
    
    # Check for text context
    text_parts = [c for c in content if c["type"] == "text"]
    assert any("X is Y" in c["text"] for c in text_parts)
    
    # Check for image context
    img_parts = [c for c in content if c["type"] == "image_url"]
    assert len(img_parts) == 1
    assert img_parts[0]["image_url"]["url"] == "data:image/png;base64,base64_fake_data"

@patch('src.rag_pipeline.retrieval_multimodal')
@patch('src.rag_pipeline.create_multimodal_message')
def test_pipeline_flow(mock_create_msg, mock_retrieve):
    """Test the full pipeline orchestrator."""
    
    # Setup Mocks
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "This is the AI answer"
    
    mock_retrieve.return_value = [Document(page_content="test")]
    mock_create_msg.return_value = "Mocked Human Message"
    
    # Run
    response = multimodal_rag_pipeline("query", "mock_vs", "mock_img_store", mock_llm)
    
    # Assert
    assert response == "This is the AI answer"
    mock_retrieve.assert_called_once()
    mock_llm.invoke.assert_called_once()