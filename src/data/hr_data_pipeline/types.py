"""Type definitions for HR data structures."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Experience:
    """Represents a work experience entry."""
    company: str
    role: str
    duration: str
    responsibilities: List[str]


@dataclass
class SkillSet:
    """Represents a set of skills in a specific category."""
    category: str
    skills: List[str]


@dataclass
class QAData:
    """Represents a question-answer pair with context."""
    question: str
    answer: str
    context: str


@dataclass
class BasicInfo:
    """Represents basic profile information."""
    name: str
    location: str
    current_role: str


@dataclass
class HRProfile:
    """Represents a complete HR profile."""
    id: str
    basic_info: BasicInfo
    experience: List[Experience]
    skills: List[SkillSet]
    qa_data: List[QAData]


@dataclass
class ProcessedChunk:
    """Represents a processed chunk of text with metadata."""
    text: str
    embedding: List[float]
    metadata: dict
    profile_id: str
    chunk_id: str
    source_type: str  # e.g., 'experience', 'qa', 'skills'


@dataclass
class OpenSearchDocument:
    """Represents a document to be indexed in OpenSearch."""
    id: str
    text: str
    embedding: List[float]
    profile_id: str
    chunk_id: str
    source_type: str
    metadata: dict


@dataclass
class ProcessingResult:
    """Represents the result of processing an HR profile."""
    profile_id: str
    chunks: List[ProcessedChunk]
    error: Optional[str] = None
    success: bool = True
