#!/usr/bin/env python3
"""Script to test the OpenSearch indexing pipeline."""

import json
import logging
import uuid
from pathlib import Path

from hr_data_pipeline.processor import HRDataProcessor
from hr_data_pipeline.config import opensearch_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_profile():
    """Load a sample profile for testing."""
    sample_path = Path(__file__).parent / "sample_data" / "profile1.json"
    with open(sample_path, "r") as f:
        return json.load(f)

def test_index_setup():
    """Test OpenSearch index setup."""
    processor = HRDataProcessor()
    
    logger.info("Testing index setup...")
    try:
        # Delete index if it exists (for clean testing)
        if processor.opensearch_client.indices.exists(
            index=opensearch_config.index_name
        ):
            processor.opensearch_client.indices.delete(
                index=opensearch_config.index_name
            )
            logger.info("Deleted existing index")
        
        # Set up new index
        processor.setup_opensearch_index()
        logger.info("Successfully created index with mappings")
        
        # Verify index exists
        assert processor.opensearch_client.indices.exists(
            index=opensearch_config.index_name
        ), "Index was not created"
        
        # Verify mappings
        mappings = processor.opensearch_client.indices.get_mapping(
            index=opensearch_config.index_name
        )
        assert opensearch_config.index_name in mappings, "Index not found in mappings"
        
        index_mappings = mappings[opensearch_config.index_name]['mappings']
        assert 'embedding' in index_mappings['properties'], "Embedding field not in mappings"
        assert index_mappings['properties']['embedding']['type'] == 'knn_vector', "Incorrect embedding type"
        
        logger.info("Index setup verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Index setup test failed: {str(e)}")
        return False

def test_profile_indexing():
    """Test profile processing and indexing."""
    processor = HRDataProcessor()
    
    logger.info("Testing profile indexing...")
    try:
        # Load and process sample profile
        profile_data = load_sample_profile()
        profile_data['id'] = str(uuid.uuid4())  # Ensure unique ID
        
        result = processor.process_profile(profile_data)
        assert result.success, f"Profile processing failed: {result.error}"
        logger.info(f"Successfully processed profile {result.profile_id}")
        
        # Verify chunks were created
        assert len(result.chunks) > 0, "No chunks were created"
        logger.info(f"Created {len(result.chunks)} chunks")
        
        # Verify chunks were indexed
        for chunk in result.chunks:
            response = processor.opensearch_client.get(
                index=opensearch_config.index_name,
                id=chunk.chunk_id
            )
            assert response['found'], f"Chunk {chunk.chunk_id} not found in index"
            
            # Verify chunk data
            source = response['_source']
            assert len(source['embedding']) > 0, "Embedding is empty"
            assert source['text'] == chunk.text, "Text mismatch"
            assert source['profile_id'] == result.profile_id, "Profile ID mismatch"
            assert source['source_type'] in ['overview', 'experience', 'skills', 'qa'], "Invalid source type"
            
        logger.info("Successfully verified all chunks in index")
        return True
        
    except Exception as e:
        logger.error(f"Profile indexing test failed: {str(e)}")
        return False

def main():
    """Run indexing pipeline tests."""
    logger.info("Starting indexing pipeline tests...")
    
    # Test index setup
    if not test_index_setup():
        logger.error("Index setup test failed")
        return
    
    # Test profile indexing
    if not test_profile_indexing():
        logger.error("Profile indexing test failed")
        return
    
    logger.info("All indexing pipeline tests completed successfully")

if __name__ == "__main__":
    main()
