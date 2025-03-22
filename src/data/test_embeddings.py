"""Test embeddings generation functionality."""

import pytest
from hr_data_pipeline.processor import HRDataProcessor

def test_embedding_generation():
    """Test that embeddings are generated correctly."""
    processor = HRDataProcessor()
    
    # Test with a sample text
    sample_text = "Software engineer with 5 years of experience in Python development"
    embedding = processor._generate_embedding(sample_text)
    
    # Verify embedding structure
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 1536, "Embedding should have 1536 dimensions"
    assert all(isinstance(x, float) for x in embedding), "All values should be floats"
    
    # Test with empty text
    with pytest.raises(Exception):
        processor._generate_embedding("")
    
    # Test with very long text
    long_text = "test " * 1000
    long_embedding = processor._generate_embedding(long_text)
    assert len(long_embedding) == 1536, "Long text should still produce correct dimensions"

def test_embedding_consistency():
    """Test that same input produces consistent embeddings."""
    processor = HRDataProcessor()
    text = "Test consistency of embeddings"
    
    embedding1 = processor._generate_embedding(text)
    embedding2 = processor._generate_embedding(text)
    
    # Verify they're the same
    assert embedding1 == embedding2, "Same input should produce same embedding"

def test_error_handling():
    """Test error handling in embedding generation."""
    processor = HRDataProcessor()
    
    # Test with None
    with pytest.raises(Exception):
        processor._generate_embedding(None)
    
    # Test with non-string input
    with pytest.raises(Exception):
        processor._generate_embedding(123)
