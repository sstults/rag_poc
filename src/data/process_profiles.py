#!/usr/bin/env python3
"""Script to process HR profiles and index them in OpenSearch."""

import json
import logging
from pathlib import Path

from hr_data_pipeline.processor import HRDataProcessor
from hr_data_pipeline.config import opensearch_config

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Process sample profiles and index them in OpenSearch."""
    try:
        # Initialize processor
        processor = HRDataProcessor()
        
        # Set up OpenSearch index
        logger.info("Starting profile processing pipeline")
        logger.info("Setting up OpenSearch index...")
        logger.info("OpenSearch connection details: host=%s, port=%s", 
                   opensearch_config.host, opensearch_config.port)
        processor.setup_opensearch_index()
        
        # Process sample profile
        sample_path = Path(__file__).parent / "sample_data" / "profile1.json"
        logger.info("Reading sample profile from %s", sample_path)
        
        with open(sample_path) as f:
            profile_data = json.load(f)
        
        result = processor.process_profile(profile_data)
        
        if result.success:
            logger.info("Successfully processed profile %s", result.profile_id)
            logger.info("Created %d chunks:", len(result.chunks))
            for chunk in result.chunks:
                logger.info("  - %s: %s", chunk.source_type, chunk.text[:100] + "...")
        else:
            logger.error(f"Failed to process profile: {result.error}")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
