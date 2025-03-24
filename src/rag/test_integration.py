"""Tests for the integration module."""

import json
import pytest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from opensearchpy import RequestsHttpConnection

from .config import RAGConfig, AWSConfig, OpenSearchConfig, LogConfig
from .integration import ServiceManager, with_error_handling


@pytest.fixture
def mock_opensearch():
    """Create a mock OpenSearch client."""
    return Mock()


@pytest.fixture
def mock_bedrock():
    """Create a mock Bedrock client."""
    return Mock()


@pytest.fixture
def mock_cloudwatch():
    """Create a mock CloudWatch client."""
    return Mock()


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return RAGConfig(
        aws=AWSConfig(
            region="us-east-1",
            bedrock_model_id="test-model",
            log_group="/test/logs",
            log_stream_prefix="test-"
        ),
        opensearch=OpenSearchConfig(
            host="localhost",
            port=9200,
            index_name="test-index"
        ),
        logging=LogConfig(
            level="INFO",
            enable_cloudwatch=True
        )
    )


@pytest.fixture
def service_manager(mock_opensearch, mock_bedrock, mock_cloudwatch, test_config):
    """Create a ServiceManager instance with mock services."""
    with patch('rag.integration.config', test_config):
        manager = ServiceManager()
        manager._opensearch = mock_opensearch
        manager._bedrock = mock_bedrock
        manager._cloudwatch = mock_cloudwatch
        return manager


def test_opensearch_connection(service_manager, mock_opensearch):
    """Test OpenSearch connection."""
    assert service_manager.opensearch == mock_opensearch
    
    # Test search functionality
    mock_opensearch.search.return_value = {"hits": {"hits": []}}
    
    result = service_manager.search_opensearch(
        body={"query": {"match_all": {}}}
    )
    
    assert result == {"hits": {"hits": []}}
    mock_opensearch.search.assert_called_once()


def test_bedrock_invocation(service_manager, mock_bedrock, test_config):
    """Test Bedrock model invocation."""
    mock_response = {
        "body": Mock(
            read=lambda: json.dumps({"response": "test response"})
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    
    with patch('rag.integration.config', test_config):
        result = service_manager.invoke_bedrock(
            prompt="test prompt",
            temperature=0.7
        )
    
    assert result == {"response": "test response"}
    mock_bedrock.invoke_model.assert_called_once()
    
    # Verify request parameters
    call_args = mock_bedrock.invoke_model.call_args[1]
    assert call_args["modelId"] == "test-model"
    assert "test prompt" in call_args["body"]


def test_cloudwatch_metrics(service_manager, mock_cloudwatch):
    """Test CloudWatch metrics."""
    service_manager.put_metric(
        metric_name="TestMetric",
        value=1.0,
        unit="Count",
        dimensions={"TestDim": "Value"}
    )
    
    mock_cloudwatch.put_metric_data.assert_called_once()
    
    # Verify metric data
    call_args = mock_cloudwatch.put_metric_data.call_args[1]
    assert call_args["Namespace"] == "RAG/Production"
    metric_data = call_args["MetricData"][0]
    assert metric_data["MetricName"] == "TestMetric"
    assert metric_data["Value"] == 1.0
    assert metric_data["Unit"] == "Count"
    assert metric_data["Dimensions"] == [{"Name": "TestDim", "Value": "Value"}]


def test_error_handling_decorator():
    """Test error handling decorator."""
    mock_func = Mock()
    mock_func.side_effect = [
        ClientError(
            error_response={"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            operation_name="test"
        ),
        "success"
    ]
    
    @with_error_handling(max_retries=1, retry_delay=0)
    def test_func():
        return mock_func()
    
    result = test_func()
    assert result == "success"
    assert mock_func.call_count == 2


def test_error_handling_max_retries():
    """Test error handling with max retries exceeded."""
    mock_func = Mock()
    mock_func.side_effect = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        operation_name="test"
    )
    
    @with_error_handling(max_retries=2, retry_delay=0)
    def test_func():
        return mock_func()
    
    with pytest.raises(ClientError):
        test_func()
    
    assert mock_func.call_count == 3  # Initial attempt + 2 retries


def test_validation_error_no_retry():
    """Test that validation errors are not retried."""
    mock_func = Mock()
    mock_func.side_effect = ClientError(
        error_response={"Error": {"Code": "ValidationError", "Message": "Invalid input"}},
        operation_name="test"
    )
    
    @with_error_handling(max_retries=2, retry_delay=0)
    def test_func():
        return mock_func()
    
    with pytest.raises(ClientError):
        test_func()
    
    assert mock_func.call_count == 1  # No retries


def test_service_initialization():
    """Test service initialization."""
    test_config = RAGConfig(
        aws=AWSConfig(region="us-east-1"),
        opensearch=OpenSearchConfig(host="localhost", port=9200),
        logging=LogConfig()
    )
    
    with patch('rag.integration.config', test_config), \
         patch('rag.integration.OpenSearch') as mock_opensearch_class, \
         patch('rag.integration.boto3.client') as mock_boto3:
        
        # Create manager and access properties to trigger initialization
        manager = ServiceManager()
        _ = manager.opensearch  # Force initialization
        _ = manager.bedrock
        _ = manager.cloudwatch
        
        # Verify OpenSearch client was created
        mock_opensearch_class.assert_called_once_with(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
        
        # Verify AWS clients were created
        assert mock_boto3.call_count == 2
        mock_boto3.assert_any_call('bedrock-runtime', region_name='us-east-1')
        mock_boto3.assert_any_call('cloudwatch', region_name='us-east-1')


def test_metrics_disabled(mock_cloudwatch):
    """Test metrics when disabled."""
    with patch('rag.integration.config.enable_metrics', False):
        manager = ServiceManager()
        manager._cloudwatch = mock_cloudwatch
        
        manager.put_metric(
            metric_name="TestMetric",
            value=1.0
        )
        
        mock_cloudwatch.put_metric_data.assert_not_called()


def test_custom_model_id(service_manager, mock_bedrock):
    """Test Bedrock invocation with custom model ID."""
    mock_response = {
        "body": Mock(
            read=lambda: json.dumps({"response": "test response"})
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    
    service_manager.invoke_bedrock(
        prompt="test",
        model_id="custom-model"
    )
    
    call_args = mock_bedrock.invoke_model.call_args[1]
    assert call_args["modelId"] == "custom-model"


def test_opensearch_custom_index(service_manager, mock_opensearch):
    """Test OpenSearch with custom index."""
    service_manager.search_opensearch(
        index="custom-index",
        body={"query": {"match_all": {}}}
    )
    
    call_args = mock_opensearch.search.call_args[1]
    assert call_args["index"] == "custom-index"
