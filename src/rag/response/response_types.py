"""Type definitions for response generation."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from ..retrieval.retrieval_types import SearchResult


class ResponseValidation(Enum):
    """Types of response validation checks."""
    FACTUAL = "factual"  # Check if response is supported by context
    COHERENT = "coherent"  # Check if response is logically coherent
    RELEVANT = "relevant"  # Check if response answers the question
    COMPLETE = "complete"  # Check if response is complete
    SAFE = "safe"  # Check for harmful/sensitive content


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_type: ResponseValidation
    passed: bool
    score: float  # 0.0 to 1.0
    details: str


@dataclass
class Citation:
    """Citation linking response to source chunks."""
    text: str
    chunk_id: str
    profile_id: str
    relevance_score: float


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    text: str
    citations: List[Citation]
    context_used: List[SearchResult]
    validation_results: List[ValidationResult]
    is_valid: bool
    confidence_score: float
    metadata: Dict[str, Any]


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    max_length: int = 1000
    min_confidence: float = 0.7
    require_citations: bool = True
    validation_types: List[ResponseValidation] = None

    def __post_init__(self):
        """Set default validation types if none provided."""
        if self.validation_types is None:
            self.validation_types = [
                ResponseValidation.FACTUAL,
                ResponseValidation.COHERENT,
                ResponseValidation.RELEVANT,
            ]


@dataclass
class HRPromptTemplate:
    """Template for HR-specific response generation."""
    template: str
    input_variables: List[str]
    validation_prompt: Optional[str] = None
    citation_prompt: Optional[str] = None

    @classmethod
    def default(cls) -> 'HRPromptTemplate':
        """Create default HR prompt template."""
        return cls(
            template="""Given the following context about HR profiles, answer the question
            while adhering to these guidelines:
            1. Be concise and professional
            2. Focus on factual information from the context
            3. Include relevant experience and skills
            4. Avoid speculation or unsupported claims
            5. Respect privacy and sensitive information

            Context:
            {context}

            Question:
            {question}

            Answer:""",
            input_variables=["context", "question"],
            validation_prompt="""Given the generated response and original context,
            evaluate if the response:
            1. Contains only facts supported by the context
            2. Directly answers the question
            3. Is logically coherent and complete
            4. Avoids sensitive or private information

            Response:
            {response}

            Context:
            {context}

            Question:
            {question}

            Evaluation (return as JSON):
            {
              "factual": {"passed": true/false, "score": 0.0-1.0, "details": "..."},
              "relevant": {"passed": true/false, "score": 0.0-1.0, "details": "..."},
              "coherent": {"passed": true/false, "score": 0.0-1.0, "details": "..."},
              "safe": {"passed": true/false, "score": 0.0-1.0, "details": "..."}
            }""",
            citation_prompt="""For the following response, identify the specific parts
            of the context that support each claim. Return the citations in JSON format
            with relevance scores.

            Response:
            {response}

            Context chunks:
            {context_chunks}

            Citations (return as JSON):
            {
              "citations": [
                {
                  "text": "...",
                  "chunk_id": "...",
                  "profile_id": "...",
                  "relevance_score": 0.0-1.0
                }
              ]
            }"""
        )
