#!/usr/bin/env python3
"""Script to test semantic search queries against indexed HR profiles."""

import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from hr_data_pipeline.config import opensearch_config
from hr_data_pipeline.processor import HRDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def search_profiles(client, query_text: str, k: int = 3):
    """
    Perform hybrid search using both text and vector similarity.
    
    Args:
        client: OpenSearch client
        query_text: Query text to search for
        k: Number of results to return
    """
    # Generate embedding for the query
    processor = HRDataProcessor()
    query_embedding = processor._generate_embedding(query_text)
    
    # Hybrid search query combining text and vector search
    search_query = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    # Text search component
                    {
                        "match": {
                            "text": {
                                "query": query_text,
                                "boost": 0.3  # Weight for text search
                            }
                        }
                    },
                    # Vector similarity component
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k,
                                "boost": 0.7  # Weight for vector search
                            }
                        }
                    }
                ]
            }
        }
    }
    
    try:
        response = client.search(
            body=search_query,
            index=opensearch_config.index_name
        )
        return response['hits']['hits']
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise

def main():
    """Run test queries against the indexed profiles."""
    # Initialize OpenSearch client
    client = OpenSearch(
        hosts=[{
            'host': opensearch_config.host,
            'port': opensearch_config.port
        }],
        http_auth=None,  # Security disabled for development
        use_ssl=opensearch_config.use_ssl,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    
    # Test queries
    test_queries = [
        "What cloud technologies does the candidate know?",
        "Tell me about the candidate's mentoring experience",
        "What are the candidate's main programming languages?",
        "Experience with web development",
        "Database optimization experience",
    ]
    
    logger.info("Starting semantic search tests...")
    
    for query in test_queries:
        logger.info("\nQuery: %s", query)
        results = search_profiles(client, query)
        
        logger.info("Top results:")
        for i, hit in enumerate(results, 1):
            score = hit['_score']
            text = hit['_source']['text']
            source_type = hit['_source']['source_type']
            logger.info(f"\n{i}. Score: {score:.4f}")
            logger.info(f"Type: {source_type}")
            logger.info(f"Text: {text[:200]}...")

if __name__ == "__main__":
    main()
