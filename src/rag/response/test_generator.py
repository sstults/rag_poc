"""Tests for the response generation module."""

import json
import pytest
from unittest.mock import Mock, patch

from langchain_community.llms import Bedrock

from ..question_processing.question_processing_types import ProcessedQuestion, Entity, EntityType
from ..retrieval.retrieval_types import RetrievalResult, SearchResult, RetrievalStrategy
from .generator import ResponseGenerator
from .response_types import (
    HRPromptTemplate,
    ResponseConfig,
    ResponseValidation,
)


@pytest.fixture
def mock_llm():
    """Create a mock Bedrock LLM."""
    llm = Mock(spec=Bedrock)
    llm.model_id = "anthropic.claude-v2"
    
    # Create a mock for the invoke method
    mock_invoke = Mock()
    
    # Set up the return values for different prompts
    validation_json = {
        "factual": {"passed": True, "score": 0.9, "details": "All claims supported"},
        "relevant": {"passed": True, "score": 0.85, "details": "Answers question"},
        "coherent": {"passed": True, "score": 0.95, "details": "Well structured"},
        "safe": {"passed": True, "score": 1.0, "details": "No sensitive info"}
    }
    
    citation_json = {
        "citations": [
            {
                "text": "Senior Python developer with AWS experience",
                "chunk_id": "chunk1",
                "profile_id": "profile1",
                "relevance_score": 0.9
            }
        ]
    }
    
    # Configure the mock to return different values based on the input
    def side_effect(prompt):
        if "evaluate if the response" in prompt:
            return json.dumps(validation_json)
        elif "identify the specific parts" in prompt:
            return json.dumps(citation_json)
        else:
            return "The candidate is a senior Python developer with extensive AWS experience."
    
    mock_invoke.side_effect = side_effect
    llm.invoke = mock_invoke
    
    return llm


@pytest.fixture
def sample_question():
    """Create a sample processed question."""
    return ProcessedQuestion(
        original_text="What experience does the candidate have with Python and AWS?",
        normalized_text="What experience does the candidate have with Python and AWS?",
        entities=[
            Entity(
                type=EntityType.SKILL,
                value="Python",
                start=39,
                end=45
            ),
            Entity(
                type=EntityType.CERTIFICATION,
                value="AWS",
                start=50,
                end=53
            )
        ],
        validation_errors=[],
        is_valid=True,
        formulated_query="Python AWS experience technical expertise cloud"
    )


@pytest.fixture
def sample_retrieval_result():
    """Create a sample retrieval result."""
    return RetrievalResult(
        results=[
            SearchResult(
                text="Senior Python developer with AWS experience",
                score=0.9,
                metadata={"role": "developer", "skill": "Python"},
                source_type="experience",
                profile_id="profile1",
                chunk_id="chunk1"
            ),
            SearchResult(
                text="Led Python development team on cloud projects",
                score=0.7,
                metadata={"role": "team lead", "skill": "Python"},
                source_type="experience",
                profile_id="profile1",
                chunk_id="chunk2"
            )
        ],
        strategy_used=RetrievalStrategy.HYBRID,
        total_found=2,
        execution_time=0.1,
        query_entities=[],
        context_window="Senior Python developer with AWS experience\n\n"
        "Led Python development team on cloud projects"
    )


@pytest.fixture
def generator(mock_llm):
    """Create a ResponseGenerator instance."""
    return ResponseGenerator(mock_llm)


def test_response_generation(generator, sample_question, sample_retrieval_result):
    """Test basic response generation."""
    response = generator.generate(sample_question, sample_retrieval_result)

    assert response.text
    assert response.is_valid
    assert response.confidence_score > generator.config.min_confidence
    assert len(response.validation_results) > 0
    assert len(response.citations) > 0

    # Check metadata
    assert "generation_time" in response.metadata
    assert response.metadata["model_id"] == "anthropic.claude-v2"
    assert response.metadata["retrieval_strategy"] == "hybrid"


def test_custom_prompt_template(mock_llm, sample_question, sample_retrieval_result):
    """Test response generation with custom prompt template."""
    custom_template = HRPromptTemplate(
        template="Q: {question}\nContext: {context}\nA:",
        input_variables=["question", "context"],
        validation_prompt=HRPromptTemplate.default().validation_prompt,
        citation_prompt=HRPromptTemplate.default().citation_prompt
    )

    generator = ResponseGenerator(mock_llm, prompt_template=custom_template)
    response = generator.generate(sample_question, sample_retrieval_result)

    assert response.text
    assert response.is_valid


def test_response_validation(generator, sample_question, sample_retrieval_result):
    """Test response validation."""
    response = generator.generate(sample_question, sample_retrieval_result)

    # Check validation results
    assert len(response.validation_results) > 0
    for result in response.validation_results:
        assert result.check_type in ResponseValidation
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 1
        assert result.details


def test_citation_extraction(generator, sample_question, sample_retrieval_result):
    """Test citation extraction."""
    response = generator.generate(sample_question, sample_retrieval_result)

    assert len(response.citations) > 0
    for citation in response.citations:
        assert citation.text
        assert citation.chunk_id
        assert citation.profile_id
        assert 0 <= citation.relevance_score <= 1


def test_confidence_scoring(generator, sample_question, sample_retrieval_result):
    """Test confidence score calculation."""
    response = generator.generate(sample_question, sample_retrieval_result)

    assert 0 <= response.confidence_score <= 1

    # Higher scores for passing validations
    assert response.confidence_score > 0.8  # Since mock validation returns high scores


def test_custom_config(mock_llm, sample_question, sample_retrieval_result):
    """Test response generation with custom configuration."""
    config = ResponseConfig(
        max_length=500,
        min_confidence=0.8,
        require_citations=True,
        validation_types=[
            ResponseValidation.FACTUAL,
            ResponseValidation.RELEVANT
        ]
    )

    generator = ResponseGenerator(mock_llm, config=config)
    response = generator.generate(sample_question, sample_retrieval_result)

    # Check only requested validation types were used
    validation_types = {r.check_type for r in response.validation_results}
    assert validation_types == {
        ResponseValidation.FACTUAL,
        ResponseValidation.RELEVANT
    }


def test_failed_validation(mock_llm, sample_question, sample_retrieval_result):
    """Test handling of failed validation."""
    # Mock LLM to return failed validation
    def mock_invoke(prompt):
        if "evaluate if the response" in prompt:
            return '{"factual": {"passed": false, "score": 0.3, "details": "Unsupported claims"}, "relevant": {"passed": true, "score": 0.8, "details": "Answers question"}}'
        elif "identify the specific parts" in prompt:  # Citation prompt
            return '{"citations": []}'
        else:
            return "Generated response"

    mock_llm.invoke = Mock(side_effect=mock_invoke)
    mock_llm.model_id = "anthropic.claude-v2"

    generator = ResponseGenerator(mock_llm)
    response = generator.generate(sample_question, sample_retrieval_result)

    assert not response.is_valid
    assert response.confidence_score < generator.config.min_confidence


def test_failed_citation_extraction(mock_llm, sample_question, sample_retrieval_result):
    """Test handling of failed citation extraction."""
    # Mock LLM to return invalid JSON for citations
    def mock_invoke(prompt):
        if "identify the specific parts" in prompt:
            return "invalid json"
        elif "evaluate if the response" in prompt:  # Validation prompt
            return '{"factual": {"passed": true, "score": 0.9, "details": "All claims supported"}, "relevant": {"passed": true, "score": 0.85, "details": "Answers question"}}'
        else:
            return "Generated response"

    mock_llm.invoke = Mock(side_effect=mock_invoke)
    mock_llm.model_id = "anthropic.claude-v2"

    generator = ResponseGenerator(mock_llm)
    response = generator.generate(sample_question, sample_retrieval_result)

    # Should still generate response but without citations
    assert response.text
    assert len(response.citations) == 0


def test_context_formatting(generator):
    """Test context formatting from search results."""
    results = [
        SearchResult(
            text="Experience with Python",
            score=0.9,
            metadata={"role": "developer", "skill": "Python"},
            source_type="experience",
            profile_id="profile1",
            chunk_id="chunk1"
        )
    ]

    context = generator._format_context(results)

    # Check metadata inclusion
    assert "[EXPERIENCE]" in context
    assert "Role: developer" in context
    assert "Skills: Python" in context
    assert "Experience with Python" in context


def test_empty_results(generator, sample_question):
    """Test handling of empty retrieval results."""
    empty_result = RetrievalResult(
        results=[],
        strategy_used=RetrievalStrategy.HYBRID,
        total_found=0,
        execution_time=0.1,
        query_entities=[],
        context_window=None
    )

    response = generator.generate(sample_question, empty_result)

    # Should still generate a response
    assert response.text
    assert len(response.context_used) == 0
    assert len(response.citations) == 0
