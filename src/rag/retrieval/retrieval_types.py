"""Type definitions for retrieval pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from ..question_processing.question_processing_types import Entity


class RetrievalStrategy(Enum):
    """Types of retrieval strategies."""
    SEMANTIC = "semantic"  # Pure vector similarity
    KEYWORD = "keyword"   # Pure keyword/BM25
    HYBRID = "hybrid"     # Combination of semantic and keyword


@dataclass
class SearchResult:
    """Single search result from retrieval."""
    text: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # overview, experience, skills, qa
    profile_id: str
    chunk_id: str


@dataclass
class RetrievalResult:
    """Results from retrieval pipeline."""
    results: List[SearchResult]
    strategy_used: RetrievalStrategy
    total_found: int
    execution_time: float
    query_entities: List[Entity]
    context_window: Optional[str] = None  # Optimized context for response generation


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    index_name: str
    semantic_weight: float = 0.7  # Weight for semantic search in hybrid
    keyword_weight: float = 0.3   # Weight for keyword search in hybrid
    max_results: int = 10
    min_score: float = 0.1
    context_window_size: int = 2000  # Characters
