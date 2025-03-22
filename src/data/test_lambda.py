#!/usr/bin/env python3
"""Tests for the Lambda function handler."""

import json
import logging
import unittest
from unittest.mock import patch, MagicMock

from lambda_function import lambda_handler
from hr_data_pipeline.types import ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLambdaHandler(unittest.TestCase):
    """Test cases for the Lambda function handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_event = {
            'Records': [{
                'eventSource': 'aws:s3',
                's3': {
                    'bucket': {
                        'name': 'test-bucket'
                    },
                    'object': {
                        'key': 'profiles/test.json'
                    }
                }
            }]
        }
        
        self.sample_context = MagicMock()
        
        # Sample successful processing result
        self.sample_result = ProcessingResult(
            profile_id="test-123",
            chunks=[MagicMock() for _ in range(3)],
            success=True
        )

    @patch('lambda_function.aws_config')
    @patch('lambda_function.processor')
    def test_successful_processing(self, mock_processor, mock_config):
        """Test successful processing of an S3 event."""
        # Configure mocks
        mock_config.s3_bucket = 'test-bucket'
        mock_processor.process_s3_file.return_value = self.sample_result
        
        # Call handler
        response = lambda_handler(self.sample_event, self.sample_context)
        
        # Verify response
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertEqual(body['profile_id'], 'test-123')
        self.assertEqual(body['chunks_created'], 3)
        
        # Verify processor called correctly
        mock_processor.process_s3_file.assert_called_once_with(
            'profiles/test.json'
        )

    @patch('lambda_function.aws_config')
    @patch('lambda_function.processor')
    def test_processing_failure(self, mock_processor, mock_config):
        """Test handling of processing failures."""
        # Configure mocks
        mock_config.s3_bucket = 'test-bucket'
        mock_processor.process_s3_file.return_value = None
        
        # Call handler
        response = lambda_handler(self.sample_event, self.sample_context)
        
        # Verify error response
        self.assertEqual(response['statusCode'], 500)
        body = json.loads(response['body'])
        self.assertIn('error', body)

    @patch('lambda_function.aws_config')
    def test_invalid_event(self, mock_config):
        """Test handling of invalid events."""
        # Test event without Records
        invalid_event = {}
        response = lambda_handler(invalid_event, self.sample_context)
        
        self.assertEqual(response['statusCode'], 500)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertIn('Records', body['error'])

    @patch('lambda_function.aws_config')
    def test_wrong_bucket(self, mock_config):
        """Test handling of events from wrong bucket."""
        # Configure mock
        mock_config.s3_bucket = 'expected-bucket'
        
        # Create event with wrong bucket
        wrong_bucket_event = {
            'Records': [{
                'eventSource': 'aws:s3',
                's3': {
                    'bucket': {
                        'name': 'wrong-bucket'
                    },
                    'object': {
                        'key': 'test.json'
                    }
                }
            }]
        }
        
        # Call handler
        response = lambda_handler(wrong_bucket_event, self.sample_context)
        
        # Verify error response
        self.assertEqual(response['statusCode'], 500)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertIn('Unexpected bucket', body['error'])

if __name__ == '__main__':
    unittest.main()
