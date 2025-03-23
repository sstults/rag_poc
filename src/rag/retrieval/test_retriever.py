"""Tests for the retrieval pipeline."""

import pytest
from unittest.mock import Mock, patch
from langchain_community.embeddings import BedrockEmbeddings
from opensearchpy import OpenSearch

from ..question_processing.question_processing_types import ProcessedQuestion, Entity, EntityType
from .retriever import HybridRetriever
from .retrieval_types import RetrievalConfig, RetrievalStrategy


@pytest.fixture
def mock_opensearch():
    """Create a mock OpenSearch client."""
    client = Mock(spec=OpenSearch)

    # Mock search responses
    def mock_search(index, body):
        if "knn" in body["query"]:  # Semantic search
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk1",
                            "_score": 0.9,
                            "_source": {
                                "text": "Senior Python developer with AWS experience",
                                "metadata": {"role": "developer", "skill": "Python"},
                                "source_type": "experience",
                                "profile_id": "profile1",
                            }
                        },
                        {
                            "_id": "chunk2",
                            "_score": 0.7,
                            "_source": {
                                "text": "Led Python development team on cloud projects",
                                "metadata": {"role": "team lead", "skill": "Python"},
                                "source_type": "experience",
                                "profile_id": "profile1",
                            }
                        }
                    ]
                }
            }
        else:  # Keyword search
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk3",
                            "_score": 0.8,
                            "_source": {
                                "text": "Experienced Python developer with leadership",
                                "metadata": {"role": "developer", "skill": "Python"},
                                "source_type": "overview",
                                "profile_id": "profile2",
                            }
                        },
                        {
                            "_id": "chunk1",  # Same as in semantic search
                            "_score": 0.6,
                            "_source": {
                                "text": "Senior Python developer with AWS experience",
                                "metadata": {"role": "developer", "skill": "Python"},
                                "source_type": "experience",
                                "profile_id": "profile1",
                            }
                        }
                    ]
                }
            }

    client.search = Mock(side_effect=mock_search)
    return client


@pytest.fixture
def mock_bedrock():
    """Create a mock Bedrock embeddings client."""
    embeddings = Mock(spec=BedrockEmbeddings)
    embeddings.embed_query.return_value = [0.1] * 1536  # Mock embedding vector
    return embeddings


@pytest.fixture
def config():
    """Create a retrieval configuration."""
    return RetrievalConfig(
        index_name="test-index",
        semantic_weight=0.7,
        keyword_weight=0.3,
        max_results=5,
        min_score=0.1,
        context_window_size=1000
    )


@pytest.fixture
def retriever(mock_opensearch, mock_bedrock, config):
    """Create a HybridRetriever instance."""
    return HybridRetriever(mock_opensearch, mock_bedrock, config)


@pytest.fixture
def sample_question():
    """Create a sample processed question."""
    return ProcessedQuestion(
        original_text="Find Python developers with leadership experience",
        normalized_text="Find Python developers with leadership experience",
        entities=[
            Entity(
                type=EntityType.SKILL,
                value="Python",
                start=5,
                end=11
            ),
            Entity(
                type=EntityType.SKILL,
                value="leadership",
                start=27,
                end=37
            )
        ],
        validation_errors=[],
        is_valid=True,
        formulated_query="Python developers leadership team management"
    )


def test_semantic_search(retriever, sample_question):
    """Test semantic search retrieval."""
    result = retriever.retrieve(sample_question, RetrievalStrategy.SEMANTIC)

    assert result.strategy_used == RetrievalStrategy.SEMANTIC
    assert result.total_found > 0
    assert all(r.score > 0 for r in result.results)

    # Verify semantic search was called
    retriever.client.search.assert_called_once()
    call_args = retriever.client.search.call_args[1]
    assert "knn" in call_args["body"]["query"]


def test_keyword_search(retriever, sample_question):
    """Test keyword search retrieval."""
    result = retriever.retrieve(sample_question, RetrievalStrategy.KEYWORD)

    assert result.strategy_used == RetrievalStrategy.KEYWORD
    assert result.total_found > 0
    assert all(r.score > 0 for r in result.results)

    # Verify keyword search was called
    retriever.client.search.assert_called_once()
    call_args = retriever.client.search.call_args[1]
    assert "bool" in call_args["body"]["query"]

    # Check entity boosting
    query = call_args["body"]["query"]["bool"]
    assert any(
        "metadata.skill" in str(clause) and "Python" in str(clause)
        for clause in query["should"]
    )


def test_hybrid_search(retriever, sample_question):
    """Test hybrid search retrieval."""
    result = retriever.retrieve(sample_question, RetrievalStrategy.HYBRID)

    assert result.strategy_used == RetrievalStrategy.HYBRID
    assert result.total_found > 0

    # Verify both search types were called
    assert retriever.client.search.call_count == 2

    # Check result merging and ranking
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)  # Results should be sorted by score

    # Verify duplicate results were merged
    chunk_ids = [r.chunk_id for r in result.results]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates


def test_context_window_optimization(retriever, sample_question):
    """Test context window optimization."""
    result = retriever.retrieve(sample_question)

    assert result.context_window is not None
    assert len(result.context_window) <= retriever.config.context_window_size

    # Context should contain highest scoring results
    for search_result in result.results[:2]:  # Check top 2 results
        assert search_result.text in result.context_window


def test_result_ranking(retriever, sample_question):
    """Test result ranking and scoring."""
    result = retriever.retrieve(sample_question)

    # Check score normalization and weighting
    assert all(0 <= r.score <= 1 for r in result.results)

    # Verify semantic and keyword weights
    semantic_weight = retriever.config.semantic_weight
    keyword_weight = retriever.config.keyword_weight
    assert abs(semantic_weight + keyword_weight - 1.0) < 1e-6

    # Check ranking order
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)


def test_empty_results(mock_opensearch, mock_bedrock, config):
    """Test handling of empty search results."""
    # Create new retriever with empty results mock
    mock_opensearch.search = Mock(return_value={"hits": {"hits": []}})
    retriever = HybridRetriever(mock_opensearch, mock_bedrock, config)

    # Create test question
    question = ProcessedQuestion(
        original_text="Find Python developers",
        normalized_text="Find Python developers",
        entities=[],
        validation_errors=[],
        is_valid=True,
        formulated_query="Python developers"
    )

    result = retriever.retrieve(question)
    assert result.total_found == 0
    assert len(result.results) == 0
    assert result.context_window is None


def test_invalid_question(mock_opensearch, mock_bedrock, config):
    """Test handling of invalid questions."""
    # Create new retriever with empty results mock
    mock_opensearch.search = Mock(return_value={"hits": {"hits": []}})
    retriever = HybridRetriever(mock_opensearch, mock_bedrock, config)

    invalid_question = ProcessedQuestion(
        original_text="",
        normalized_text="",
        entities=[],
        validation_errors=["Empty question"],
        is_valid=False,
        formulated_query=None
    )

    result = retriever.retrieve(invalid_question)
    assert result.total_found == 0
    assert len(result.results) == 0
    assert result.context_window is None


def test_result_deduplication(retriever, sample_question):
    """Test deduplication of results from semantic and keyword search."""
    result = retriever.retrieve(sample_question, RetrievalStrategy.HYBRID)

    # Check for duplicates
    chunk_ids = [r.chunk_id for r in result.results]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Verify scores were properly combined for duplicate results
    chunk1_result = next(r for r in result.results if r.chunk_id == "chunk1")
    assert chunk1_result.score > 0  # Should have combined semantic and keyword scores


def test_performance_metrics(retriever, sample_question):
    """Test performance metrics in retrieval results."""
    result = retriever.retrieve(sample_question)

    assert result.execution_time > 0
    assert isinstance(result.execution_time, float)

    # Execution time should be reasonable
    assert result.execution_time < 5.0  # Should complete within 5 seconds
