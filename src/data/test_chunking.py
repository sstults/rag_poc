"""Test the chunking functionality of the HR data pipeline."""

import json
from pathlib import Path

from hr_data_pipeline.processor import HRDataProcessor
from hr_data_pipeline.types import ProcessedChunk

def test_chunking():
    """Test the chunking strategy with a sample profile."""
    # Load sample profile
    sample_path = Path(__file__).parent / "sample_data" / "profile1.json"
    with open(sample_path) as f:
        profile_data = json.load(f)
    
    # Initialize processor
    processor = HRDataProcessor()
    
    # Process the profile
    result = processor.process_profile(profile_data)
    
    # Verify processing was successful
    assert result.success, f"Processing failed: {result.error}"
    
    # Print chunk analysis
    print("\nChunk Analysis:")
    print("-" * 80)
    
    for i, chunk in enumerate(result.chunks):
        print(f"\nChunk {i+1}/{len(result.chunks)}:")
        print(f"Source Type: {chunk.source_type}")
        print(f"Metadata: {json.dumps(chunk.metadata, indent=2)}")
        print("\nContent:")
        print("-" * 40)
        print(chunk.text)
        print("-" * 40)
        
        # Verify chunk structure
        assert isinstance(chunk, ProcessedChunk)
        assert chunk.profile_id == profile_data["id"]
        assert chunk.chunk_id.startswith(profile_data["id"])
        assert chunk.embedding is not None
        assert len(chunk.embedding) > 0
        
        # Verify metadata relationships
        if i > 0:
            assert chunk.metadata.get("previous_chunk_id") is not None
        if i < len(result.chunks) - 1:
            assert chunk.metadata.get("next_chunk_id") is not None
            
    print(f"\nTotal chunks created: {len(result.chunks)}")

if __name__ == "__main__":
    test_chunking()
