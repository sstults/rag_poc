# HR Data Pipeline

This component handles the processing and indexing of HR profile data for the RAG system. It processes JSON files containing HR profiles, generates embeddings using AWS Bedrock, and indexes the data in OpenSearch for hybrid search capabilities.

## Features

- Process HR profiles from S3 bucket
- Generate embeddings using AWS Bedrock
- Index data in OpenSearch with hybrid search support
- Configurable chunking strategy
- Comprehensive error handling and logging
- CLI tool for easy execution

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Configure environment variables:

A `.env` file is provided with default configuration. Update the values in `.env` to match your environment:

```ini
# AWS Configuration
AWS_REGION=us-east-1
HR_DATA_BUCKET=your-s3-bucket-name
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenSearch Configuration
OPENSEARCH_HOST=your-opensearch-host
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=hr-profiles
OPENSEARCH_USERNAME=your-username  # Optional
OPENSEARCH_PASSWORD=your-password  # Optional
OPENSEARCH_USE_SSL=true           # Optional

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_DIMENSION=1536
```

The package will automatically load these environment variables from the `.env` file.

## Usage

### Process Profiles

Process HR profiles from an S3 bucket:

```bash
# Process all JSON files in the bucket
process-profiles --bucket your-bucket-name --setup-index

# Process files with a specific prefix
process-profiles --bucket your-bucket-name --prefix profiles/2024/

# Use bucket from environment variable
process-profiles --setup-index
```

### Sample Data Format

HR profiles should be JSON files following this structure:

```json
{
  "id": "unique_id",
  "basic_info": {
    "name": "string",
    "location": "string",
    "current_role": "string"
  },
  "experience": [
    {
      "company": "string",
      "role": "string",
      "duration": "string",
      "responsibilities": ["string"]
    }
  ],
  "skills": [
    {
      "category": "string",
      "skills": ["string"]
    }
  ],
  "qa_data": [
    {
      "question": "string",
      "answer": "string",
      "context": "string"
    }
  ]
}
```

A sample profile is provided in `sample_data/profile1.json`.

## Development

### Project Structure

```
hr_data_pipeline/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── processor.py         # Core processing logic
└── types.py            # Type definitions

sample_data/            # Sample HR profiles for testing
process_profiles.py     # CLI entry point
requirements.txt        # Package dependencies
setup.py               # Package setup configuration
```

### Adding New Features

1. Update type definitions in `types.py`
2. Implement new processing logic in `processor.py`
3. Update configuration in `config.py` if needed
4. Add CLI options to `process_profiles.py` if required

### Testing

1. Use the sample profile:
```bash
# Copy sample profile to S3
aws s3 cp sample_data/profile1.json s3://your-bucket/profiles/

# Process the profile
process-profiles --bucket your-bucket --prefix profiles/
```

2. Monitor OpenSearch for indexed documents
3. Check CloudWatch logs for processing details

## Error Handling

The pipeline includes comprehensive error handling:

- S3 access issues
- Malformed JSON data
- OpenSearch connection problems
- Embedding generation failures
- Invalid profile structure

All errors are logged with appropriate context for debugging.
