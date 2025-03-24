"""Type definitions for question processing."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class EntityType(Enum):
    """Types of HR-specific entities that can be recognized in questions."""
    SKILL = "skill"
    ROLE = "role"
    COMPANY = "company"
    LOCATION = "location"
    EXPERIENCE_LEVEL = "experience_level"
    EDUCATION = "education"
    CERTIFICATION = "certification"


@dataclass
class Entity:
    """Recognized entity in a question."""
    type: EntityType
    value: str
    start: int  # Character position where entity starts
    end: int    # Character position where entity ends


@dataclass
class ValidationError:
    """Error found during question validation."""
    message: str
    severity: str  # 'error' or 'warning'


@dataclass
class ProcessedQuestion:
    """Question after processing and validation."""
    original_text: str
    normalized_text: str
    entities: List[Entity]
    validation_errors: List[ValidationError]
    is_valid: bool
    formulated_query: Optional[str] = None  # Enhanced query for retrieval


@dataclass
class QuestionContext:
    """Additional context for question processing."""
    previous_questions: List[str] = None  # For conversation history
    filters: dict = None  # Any active filters (e.g., by role, location)
    preferences: dict = None  # User preferences for answers
