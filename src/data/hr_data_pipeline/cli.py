#!/usr/bin/env python3
"""CLI script to process HR profiles and index them in OpenSearch."""

import argparse
import logging
import os
import sys
from typing import List

import boto3

from hr_data_pipeline.config import aws_config
from hr_data_pipeline.processor import HRDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_s3_files(bucket: str, prefix: str = "") -> List[str]:
    """List all JSON files in the S3 bucket with given prefix."""
    s3_client = boto3.client('s3', region_name=aws_config.region)
    files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        files.append(obj['Key'])
        return files
    except Exception as e:
        logger.error(f"Error listing S3 files: {str(e)}")
        raise

def main():
    """Main entry point for the profile processing script."""
    parser = argparse.ArgumentParser(
        description='Process HR profiles from S3 and index them in OpenSearch'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        help='S3 bucket name (overrides env variable)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default="",
        help='S3 prefix to filter files (optional)'
    )
    parser.add_argument(
        '--setup-index',
        action='store_true',
        help='Set up OpenSearch index before processing'
    )

    args = parser.parse_args()

    # Use provided bucket or fall back to environment variable
    bucket = args.bucket or aws_config.s3_bucket
    if not bucket:
        logger.error("No S3 bucket specified. Use --bucket or set HR_DATA_BUCKET env var")
        sys.exit(1)

    processor = HRDataProcessor()

    # Set up OpenSearch index if requested
    if args.setup_index:
        logger.info("Setting up OpenSearch index...")
        try:
            processor.setup_opensearch_index()
        except Exception as e:
            logger.error(f"Failed to set up OpenSearch index: {str(e)}")
            sys.exit(1)

    # List and process files
    try:
        files = list_s3_files(bucket, args.prefix)
        logger.info(f"Found {len(files)} JSON files to process")

        success_count = 0
        failure_count = 0

        for file_key in files:
            logger.info(f"Processing {file_key}...")
            result = processor.process_s3_file(file_key)
            
            if result and result.success:
                success_count += 1
                logger.info(
                    f"Successfully processed {file_key} - "
                    f"Created {len(result.chunks)} chunks"
                )
            else:
                failure_count += 1
                logger.error(
                    f"Failed to process {file_key}"
                    + (f": {result.error}" if result else "")
                )

        logger.info(
            f"Processing complete. "
            f"Successes: {success_count}, Failures: {failure_count}"
        )

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
