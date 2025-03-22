"""HR data processing module for chunking and embedding generation."""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any

import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection

from .config import aws_config, opensearch_config, processing_config
from .types import (
    HRProfile,
    ProcessedChunk,
    ProcessingResult,
    OpenSearchDocument,
    BasicInfo,
    Experience,
    SkillSet,
    QAData,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRDataProcessor:
    """Processes HR profiles for RAG system."""

    def __init__(self):
        """Initialize the processor with AWS and OpenSearch clients."""
        self.s3_client = boto3.client('s3', region_name=aws_config.region)
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=aws_config.region
        )
        
        # Initialize OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{
                'host': opensearch_config.host,
                'port': opensearch_config.port
            }],
            http_auth=(
                opensearch_config.username,
                opensearch_config.password
            ) if opensearch_config.username else None,
            use_ssl=opensearch_config.use_ssl,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=processing_config.chunk_size,
            chunk_overlap=processing_config.chunk_overlap,
            length_function=len,
        )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate mock embeddings for testing."""
        # For testing: generate a simple mock embedding of correct dimension
        import hashlib
        import numpy as np
        
        # Create a deterministic but seemingly random embedding from the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        
        # Generate embedding of correct dimension
        mock_embedding = list(np.random.uniform(-1, 1, processing_config.embedding_dimension))
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(mock_embedding)
        mock_embedding = [x/norm for x in mock_embedding]
        
        return mock_embedding

    def _create_chunks(self, profile: HRProfile) -> List[ProcessedChunk]:
        """Create chunks from an HR profile."""
        chunks: List[ProcessedChunk] = []
        
        # Process experience sections
        for exp in profile.experience:
            text = (
                f"Experience at {exp.company} as {exp.role} "
                f"for {exp.duration}. Responsibilities: "
                f"{'. '.join(exp.responsibilities)}"
            )
            chunk_texts = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(chunk_texts):
                chunk_id = f"{profile.id}_exp_{i}"
                embedding = self._generate_embedding(chunk_text)
                chunks.append(ProcessedChunk(
                    text=chunk_text,
                    embedding=embedding,
                    metadata={
                        "company": exp.company,
                        "role": exp.role,
                        "duration": exp.duration
                    },
                    profile_id=profile.id,
                    chunk_id=chunk_id,
                    source_type="experience"
                ))

        # Process skills
        for skill_set in profile.skills:
            text = (
                f"Skills in {skill_set.category}: "
                f"{', '.join(skill_set.skills)}"
            )
            embedding = self._generate_embedding(text)
            chunks.append(ProcessedChunk(
                text=text,
                embedding=embedding,
                metadata={"category": skill_set.category},
                profile_id=profile.id,
                chunk_id=f"{profile.id}_skills_{skill_set.category}",
                source_type="skills"
            ))

        # Process QA data
        for i, qa in enumerate(profile.qa_data):
            text = (
                f"Q: {qa.question}\nA: {qa.answer}\n"
                f"Context: {qa.context}"
            )
            embedding = self._generate_embedding(text)
            chunks.append(ProcessedChunk(
                text=text,
                embedding=embedding,
                metadata={},
                profile_id=profile.id,
                chunk_id=f"{profile.id}_qa_{i}",
                source_type="qa"
            ))

        return chunks

    def _index_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """Index processed chunks to OpenSearch."""
        for chunk in chunks:
            document = OpenSearchDocument(
                id=chunk.chunk_id,
                text=chunk.text,
                embedding=chunk.embedding,
                profile_id=chunk.profile_id,
                chunk_id=chunk.chunk_id,
                source_type=chunk.source_type,
                metadata=chunk.metadata
            )
            
            try:
                self.opensearch_client.index(
                    index=opensearch_config.index_name,
                    body=document.__dict__,
                    id=document.id,
                    refresh=True
                )
            except Exception as e:
                logger.error(f"Error indexing document {document.id}: {str(e)}")
                raise

    def process_profile(self, profile_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single HR profile."""
        try:
            # Convert raw data to HRProfile
            profile = HRProfile(
                id=profile_data.get('id', str(uuid.uuid4())),
                basic_info=BasicInfo(**profile_data['basic_info']),
                experience=[Experience(**exp) for exp in profile_data['experience']],
                skills=[SkillSet(**skill) for skill in profile_data['skills']],
                qa_data=[QAData(**qa) for qa in profile_data['qa_data']]
            )

            # Create and process chunks
            chunks = self._create_chunks(profile)
            
            # Index chunks
            self._index_chunks(chunks)

            return ProcessingResult(
                profile_id=profile.id,
                chunks=chunks,
                success=True
            )

        except Exception as e:
            logger.error(f"Error processing profile: {str(e)}")
            return ProcessingResult(
                profile_id=profile_data.get('id', 'unknown'),
                chunks=[],
                error=str(e),
                success=False
            )

    def process_s3_file(self, key: str) -> Optional[ProcessingResult]:
        """Process an HR profile JSON file from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=aws_config.s3_bucket,
                Key=key
            )
            profile_data = json.loads(response['Body'].read().decode('utf-8'))
            return self.process_profile(profile_data)
        except Exception as e:
            logger.error(f"Error processing S3 file {key}: {str(e)}")
            return None

    def setup_opensearch_index(self) -> None:
        """Set up the OpenSearch index with appropriate mappings."""
        index_body = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": processing_config.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "profile_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            }
        }

        try:
            if not self.opensearch_client.indices.exists(
                index=opensearch_config.index_name
            ):
                self.opensearch_client.indices.create(
                    index=opensearch_config.index_name,
                    body=index_body
                )
                logger.info(f"Created index {opensearch_config.index_name}")
        except Exception as e:
            logger.error(f"Error setting up OpenSearch index: {str(e)}")
            raise
