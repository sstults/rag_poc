"""Configuration module for RAG system."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-v2"
    )
    log_group: str = os.getenv(
        "CLOUDWATCH_LOG_GROUP",
        "/rag/production"
    )
    log_stream_prefix: str = os.getenv(
        "CLOUDWATCH_LOG_STREAM_PREFIX",
        "rag-"
    )


@dataclass
class OpenSearchConfig:
    """OpenSearch configuration."""
    host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    index_name: str = os.getenv("OPENSEARCH_INDEX", "hr-profiles")
    username: Optional[str] = os.getenv("OPENSEARCH_USERNAME")
    password: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    use_ssl: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    enable_cloudwatch: bool = os.getenv(
        "ENABLE_CLOUDWATCH_LOGS",
        "true"
    ).lower() == "true"


@dataclass
class RAGConfig:
    """Main RAG system configuration."""
    aws: AWSConfig
    opensearch: OpenSearchConfig
    logging: LogConfig
    
    def __init__(
        self,
        aws: Optional[AWSConfig] = None,
        opensearch: Optional[OpenSearchConfig] = None,
        logging: Optional[LogConfig] = None,
        **kwargs
    ):
        """Initialize with default configs."""
        self.aws = aws or AWSConfig()
        self.opensearch = opensearch or OpenSearchConfig()
        self.logging = logging or LogConfig()
    
    # Performance tuning
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Error handling
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
    
    # Monitoring
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_namespace: str = os.getenv("METRICS_NAMESPACE", "RAG/Production")


# Global configuration instance
config = RAGConfig()
