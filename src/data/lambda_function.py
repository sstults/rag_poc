"""AWS Lambda handler for HR data processing."""

import json
import logging
import os
from typing import Dict, Any

from hr_data_pipeline.processor import HRDataProcessor
from hr_data_pipeline.config import aws_config

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize processor
processor = HRDataProcessor()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing HR profile data from S3.
    
    Args:
        event: Lambda event containing S3 bucket and key information
        context: Lambda context
        
    Returns:
        Dict containing processing status and results
    """
    try:
        # Log the received event
        logger.info("Processing event: %s", json.dumps(event))
        
        # Extract S3 information from the event
        if 'Records' not in event:
            raise ValueError("Event does not contain S3 Records")
            
        for record in event['Records']:
            if record['eventSource'] != 'aws:s3':
                continue
                
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            logger.info(f"Processing file {key} from bucket {bucket}")
            
            # Verify bucket matches configuration
            if bucket != aws_config.s3_bucket:
                raise ValueError(
                    f"Unexpected bucket: {bucket}. "
                    f"Expected: {aws_config.s3_bucket}"
                )
            
            # Process the file
            result = processor.process_s3_file(key)
            
            if not result:
                raise Exception(f"Failed to process file {key}")
            
            if not result.success:
                raise Exception(
                    f"Processing failed for {key}: {result.error}"
                )
            
            logger.info(
                f"Successfully processed file {key}. "
                f"Created {len(result.chunks)} chunks."
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Processing completed successfully',
                    'profile_id': result.profile_id,
                    'chunks_created': len(result.chunks)
                })
            }
            
    except Exception as e:
        logger.error("Processing failed: %s", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
