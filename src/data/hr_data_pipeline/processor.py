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
        """Generate embeddings using Amazon Bedrock."""
        try:
            # Prepare the request body
            request_body = {
                "inputText": text
            }
            
            # Invoke Bedrock API
            response = self.bedrock_client.invoke_model(
                modelId=aws_config.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise ValueError("No embedding found in Bedrock response")
                
            # Verify embedding dimension
            if len(embedding) != processing_config.embedding_dimension:
                logger.warning(
                    f"Embedding dimension mismatch. Expected {processing_config.embedding_dimension}, "
                    f"got {len(embedding)}"
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _prepare_text_for_chunking(self, profile: HRProfile) -> str:
        """Prepare a coherent text representation of the profile for chunking."""
        sections = []
        
        # Basic info section
        basic_info = (
            f"Profile Overview:\n"
            f"Name: {profile.basic_info.name}\n"
            f"Current Role: {profile.basic_info.current_role}\n"
            f"Location: {profile.basic_info.location}\n"
        )
        sections.append(basic_info)
        
        # Experience section with context
        if profile.experience:
            exp_sections = []
            for exp in profile.experience:
                exp_text = (
                    f"Professional Experience:\n"
                    f"Role: {exp.role} at {exp.company}\n"
                    f"Duration: {exp.duration}\n"
                    f"Key Responsibilities:\n"
                )
                exp_text += "\n".join(f"- {resp}" for resp in exp.responsibilities)
                exp_sections.append(exp_text)
            sections.extend(exp_sections)
        
        # Skills section with categories
        if profile.skills:
            skills_sections = []
            for skill_set in profile.skills:
                skills_text = (
                    f"Professional Skills - {skill_set.category}:\n"
                    f"{', '.join(skill_set.skills)}"
                )
                skills_sections.append(skills_text)
            sections.extend(skills_sections)
        
        # QA section with context
        if profile.qa_data:
            qa_sections = []
            for qa in profile.qa_data:
                qa_text = (
                    f"Interview Q&A:\n"
                    f"Question: {qa.question}\n"
                    f"Answer: {qa.answer}\n"
                    f"Additional Context: {qa.context}"
                )
                qa_sections.append(qa_text)
            sections.extend(qa_sections)
        
        return "\n\n".join(sections)

    def _create_chunks(self, profile: HRProfile) -> List[ProcessedChunk]:
        """Create semantically meaningful chunks from an HR profile."""
        chunks: List[ProcessedChunk] = []
        
        # Prepare the complete text with proper formatting and context
        full_text = self._prepare_text_for_chunking(profile)
        
        # Split the text into chunks while preserving semantic boundaries
        chunk_texts = self.text_splitter.split_text(full_text)
        
        # Process each chunk with enhanced metadata
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{profile.id}_chunk_{i}"
            
            # Determine the primary content type and extract relevant metadata
            metadata = {
                "name": profile.basic_info.name,
                "current_role": profile.basic_info.current_role,
                "location": profile.basic_info.location,
                "chunk_index": i,
                "total_chunks": len(chunk_texts)
            }
            
            # Identify the primary content type based on the chunk content
            if "Professional Experience:" in chunk_text:
                source_type = "experience"
                # Extract company and role if present in the chunk
                for exp in profile.experience:
                    if exp.company in chunk_text and exp.role in chunk_text:
                        metadata.update({
                            "company": exp.company,
                            "role": exp.role,
                            "duration": exp.duration
                        })
                        break
            elif "Professional Skills" in chunk_text:
                source_type = "skills"
                # Extract skill category if present
                for skill_set in profile.skills:
                    if skill_set.category in chunk_text:
                        metadata["category"] = skill_set.category
                        break
            elif "Interview Q&A:" in chunk_text:
                source_type = "qa"
                # Add question context if present
                for qa in profile.qa_data:
                    if qa.question in chunk_text:
                        metadata["question"] = qa.question
                        break
            else:
                source_type = "overview"
            
            # Add relationship metadata
            if i > 0:
                metadata["previous_chunk_id"] = f"{profile.id}_chunk_{i-1}"
            if i < len(chunk_texts) - 1:
                metadata["next_chunk_id"] = f"{profile.id}_chunk_{i+1}"
            
            # Generate embedding and create chunk
            embedding = self._generate_embedding(chunk_text)
            chunks.append(ProcessedChunk(
                text=chunk_text,
                embedding=embedding,
                metadata=metadata,
                profile_id=profile.id,
                chunk_id=chunk_id,
                source_type=source_type
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
