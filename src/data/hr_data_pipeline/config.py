"""Configuration settings for the HR data pipeline."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AWSConfig:
    """AWS configuration settings."""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("HR_DATA_BUCKET")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")

@dataclass
class OpenSearchConfig:
    """OpenSearch configuration settings."""
    host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    index_name: str = os.getenv("OPENSEARCH_INDEX", "hr-profiles")
    username: Optional[str] = os.getenv("OPENSEARCH_USERNAME")
    password: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    use_ssl: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"

@dataclass
class ProcessingConfig:
    """Data processing configuration settings."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Global configuration instances
aws_config = AWSConfig()
opensearch_config = OpenSearchConfig()
processing_config = ProcessingConfig()
