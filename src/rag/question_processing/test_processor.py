"""Tests for the question processing module."""

import pytest

from .processor import QuestionProcessor
from .question_processing_types import Entity, EntityType, QuestionContext, ValidationError


@pytest.fixture
def processor():
    """Create a QuestionProcessor instance for testing."""
    return QuestionProcessor()


def test_validation_empty_question(processor):
    """Test validation of empty questions."""
    result = processor.process_question("")
    assert not result.is_valid
    assert any(e.severity == "error" for e in result.validation_errors)
    assert any(e.message == "Question cannot be empty" for e in result.validation_errors)


def test_validation_short_question(processor):
    """Test validation of too-short questions."""
    result = processor.process_question("skills?")
    assert result.is_valid  # Warnings don't make it invalid
    assert any(
        e.severity == "warning" and "too short" in e.message
        for e in result.validation_errors
    )


def test_validation_non_hr_question(processor):
    """Test validation of non-HR questions."""
    result = processor.process_question(
        "What is the weather like today?"
    )
    assert result.is_valid  # It's a warning, not an error
    assert any(
        e.severity == "warning" and "HR-related" in e.message
        for e in result.validation_errors
    )


def test_validation_harmful_content(processor):
    """Test validation of questions with harmful content."""
    result = processor.process_question(
        "What is the candidate's home address and phone number?"
    )
    assert not result.is_valid
    assert any(
        e.severity == "error" and "sensitive" in e.message
        for e in result.validation_errors
    )


def test_entity_recognition_skills(processor):
    """Test recognition of skill entities."""
    result = processor.process_question(
        "Find candidates with Python and JavaScript experience"
    )
    assert result.is_valid

    skill_entities = [e for e in result.entities if e.type == EntityType.SKILL]
    assert len(skill_entities) == 2
    assert any(e.value.lower() == "python" for e in skill_entities)
    assert any(e.value.lower() == "javascript" for e in skill_entities)


def test_entity_recognition_roles(processor):
    """Test recognition of role entities."""
    result = processor.process_question(
        "Looking for a Senior Software Engineer with leadership experience"
    )
    assert result.is_valid

    # Check role entity
    role_entities = [e for e in result.entities if e.type == EntityType.ROLE]
    assert any(e.value == "Software Engineer" for e in role_entities)

    # Check experience level entity
    level_entities = [e for e in result.entities if e.type == EntityType.EXPERIENCE_LEVEL]
    assert any(e.value == "Senior" for e in level_entities)

    # Check skill entity
    skill_entities = [e for e in result.entities if e.type == EntityType.SKILL]
    assert any(e.value == "leadership" for e in skill_entities)


def test_entity_recognition_education(processor):
    """Test recognition of education entities."""
    result = processor.process_question(
        "Find candidates with a Bachelor's in Computer Science"
    )
    assert result.is_valid

    edu_entities = [e for e in result.entities if e.type == EntityType.EDUCATION]
    assert len(edu_entities) >= 2  # Should find both "Bachelor's" and "Computer Science"


def test_query_formulation_basic(processor):
    """Test basic query formulation."""
    result = processor.process_question(
        "Find Python developers with AWS certification"
    )
    assert result.is_valid
    assert result.formulated_query is not None

    # Query should include original terms and expansions
    assert "Python" in result.formulated_query
    assert "developer" in result.formulated_query.lower()
    assert "AWS" in result.formulated_query


def test_query_formulation_with_context(processor):
    """Test query formulation with context."""
    context = QuestionContext(
        filters={"location": "remote", "experience": "senior"}
    )

    result = processor.process_question(
        "Find Python developers with AWS certification",
        context=context
    )
    assert result.is_valid
    assert result.formulated_query is not None

    # Query should include context filters
    assert "location:remote" in result.formulated_query
    assert "experience:senior" in result.formulated_query


def test_end_to_end_processing(processor):
    """Test complete question processing flow."""
    context = QuestionContext(
        filters={"location": "San Francisco, CA"}
    )

    result = processor.process_question(
        "Find senior Python developers with AWS certification and leadership experience",
        context=context
    )

    # Check validation
    assert result.is_valid
    assert not any(e.severity == "error" for e in result.validation_errors)

    # Check entity recognition
    entity_types = {e.type: e.value for e in result.entities}
    assert EntityType.SKILL in entity_types  # Python, leadership
    assert EntityType.EXPERIENCE_LEVEL in entity_types  # senior
    assert EntityType.CERTIFICATION in entity_types  # AWS

    # Check query formulation
    assert result.formulated_query is not None
    assert "Python" in result.formulated_query
    assert "senior" in result.formulated_query.lower()
    assert "AWS" in result.formulated_query
    assert "leadership" in result.formulated_query.lower()
    assert "San Francisco" in result.formulated_query


def test_normalization(processor):
    """Test text normalization."""
    result = processor.process_question(
        "  Find   Python    developers  "
    )
    assert result.is_valid
    assert result.normalized_text == "Find Python developers"


def test_multiple_entity_types(processor):
    """Test recognition of multiple entity types in same question."""
    result = processor.process_question(
        "Find Senior Software Engineer in Seattle, WA with Python and AWS experience"
    )
    assert result.is_valid

    entity_types = {e.type for e in result.entities}
    assert EntityType.EXPERIENCE_LEVEL in entity_types  # Senior
    assert EntityType.ROLE in entity_types  # Software Engineer
    assert EntityType.LOCATION in entity_types  # Seattle, WA
    assert EntityType.SKILL in entity_types  # Python
    assert EntityType.CERTIFICATION in entity_types  # AWS
