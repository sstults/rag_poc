"""Integration module for external services."""

import json
import logging
import time
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import boto3
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection
from watchtower import CloudWatchLogHandler

from .config import config

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(config.logging.level)

# Add CloudWatch handler if enabled
if config.logging.enable_cloudwatch:
    cloudwatch_handler = CloudWatchLogHandler(
        log_group_name=config.aws.log_group,
        log_stream_name=f"{config.aws.log_stream_prefix}{int(time.time())}",
        boto3_client=boto3.client(
            'logs',
            region_name=config.aws.region
        )
    )
    logger.addHandler(cloudwatch_handler)

# Type variable for generic function return type
T = TypeVar('T')


def with_error_handling(
    max_retries: int = config.max_retries,
    retry_delay: float = config.retry_delay
) -> Callable:
    """Decorator for error handling and retries.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    error_msg = e.response['Error']['Message']
                    
                    logger.error(
                        f"AWS error: {error_code} - {error_msg}",
                        extra={
                            "service": func.__module__,
                            "operation": func.__name__,
                            "attempt": attempt + 1
                        }
                    )
                    
                    # Don't retry certain errors
                    if error_code in ['ValidationError', 'InvalidRequest']:
                        raise
                    
                    last_error = e
                    
                except Exception as e:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            "service": func.__module__,
                            "operation": func.__name__,
                            "attempt": attempt + 1
                        }
                    )
                    last_error = e
                
                if attempt < max_retries:
                    sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
            
            raise last_error
            
        return wrapper
    return decorator


class ServiceManager:
    """Manages connections to external services."""
    
    def __init__(self):
        """Initialize service connections."""
        self._opensearch: Optional[OpenSearch] = None
        self._bedrock: Optional[Any] = None
        self._cloudwatch: Optional[Any] = None
        
        # Initialize services
        self.opensearch  # Initialize OpenSearch
        self.bedrock  # Initialize Bedrock
        self.cloudwatch  # Initialize CloudWatch
    
    @property
    def opensearch(self) -> OpenSearch:
        """Get OpenSearch client."""
        if not self._opensearch:
            self._opensearch = OpenSearch(
                hosts=[{
                    'host': config.opensearch.host,
                    'port': config.opensearch.port
                }],
                http_auth=(
                    config.opensearch.username,
                    config.opensearch.password
                ) if config.opensearch.username else None,
                use_ssl=config.opensearch.use_ssl,
                verify_certs=False,
                connection_class=RequestsHttpConnection,
            )
        return self._opensearch
    
    @property
    def bedrock(self) -> Any:
        """Get Bedrock client."""
        if not self._bedrock:
            self._bedrock = boto3.client(
                'bedrock-runtime',
                region_name=config.aws.region
            )
        return self._bedrock
    
    @property
    def cloudwatch(self) -> Any:
        """Get CloudWatch client."""
        if not self._cloudwatch:
            self._cloudwatch = boto3.client(
                'cloudwatch',
                region_name=config.aws.region
            )
        return self._cloudwatch
    
    @with_error_handling()
    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """Put a metric to CloudWatch.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Metric unit
            dimensions: Optional metric dimensions
        """
        if not config.enable_metrics:
            return
            
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.now(UTC)
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v}
                for k, v in dimensions.items()
            ]
        
        self.cloudwatch.put_metric_data(
            Namespace=config.metrics_namespace,
            MetricData=[metric_data]
        )
    
    @with_error_handling()
    def invoke_bedrock(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Invoke Bedrock model.
        
        Args:
            prompt: Input prompt
            model_id: Optional model ID override
            **kwargs: Additional model parameters
            
        Returns:
            Model response
        """
        model_id = model_id or config.aws.bedrock_model_id
        
        request_body = {
            "inputText": prompt,  # Match Bedrock API format
            **kwargs
        }
        
        response = self.bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        return json.loads(response['body'].read())
    
    @with_error_handling()
    def search_opensearch(
        self,
        index: Optional[str] = None,
        body: Dict[str, Any] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute OpenSearch query.
        
        Args:
            index: Optional index name override
            body: Query body
            **kwargs: Additional query parameters
            
        Returns:
            Search results
        """
        index = index or config.opensearch.index_name
        return self.opensearch.search(
            index=index,
            body=body,
            **kwargs
        )


# Global service manager instance
services = ServiceManager()
