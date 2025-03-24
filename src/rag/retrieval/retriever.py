"""Retrieval pipeline implementation using OpenSearch."""

import json
import time
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import BedrockEmbeddings
from opensearchpy import OpenSearch

from ..question_processing.question_processing_types import ProcessedQuestion, Entity
from .retrieval_types import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SearchResult,
)


class HybridRetriever:
    """Implements hybrid search retrieval using OpenSearch."""

    def __init__(
        self,
        opensearch_client: OpenSearch,
        bedrock_embeddings: BedrockEmbeddings,
        config: RetrievalConfig
    ):
        """Initialize the retriever.

        Args:
            opensearch_client: Configured OpenSearch client
            bedrock_embeddings: Bedrock embeddings client
            config: Retrieval configuration
        """
        self.client = opensearch_client
        self.embeddings = bedrock_embeddings
        self.config = config

    def _semantic_search(
        self,
        query_embedding: List[float],
        size: int = 10,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using kNN.

        Args:
            query_embedding: Query vector
            size: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of search hits
        """
        knn_query = {
            "size": size,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": size
                    }
                }
            },
            "_source": True
        }

        response = self.client.search(
            index=self.config.index_name,
            body=knn_query
        )

        hits = []
        for hit in response["hits"]["hits"]:
            if hit["_score"] >= min_score:
                hits.append(hit)

        return hits

    def _keyword_search(
        self,
        query: str,
        entities: List[Entity],
        size: int = 10,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25.

        Args:
            query: Search query
            entities: Recognized entities
            size: Number of results to return
            min_score: Minimum relevance score

        Returns:
            List of search hits
        """
        # Build entity-aware bool query
        should_clauses = []

        # Add text match on full query
        should_clauses.append({
            "match": {
                "text": {
                    "query": query,
                    "boost": 1.0
                }
            }
        })

        # Add entity-specific matches
        for entity in entities:
            field_boost = {
                "skill": 2.0,
                "role": 1.5,
                "company": 1.2,
                "experience_level": 1.3,
                "education": 1.2,
                "certification": 1.4
            }.get(entity.type.value, 1.0)

            should_clauses.append({
                "match": {
                    f"metadata.{entity.type.value}": {
                        "query": entity.value,
                        "boost": field_boost
                    }
                }
            })

        bool_query = {
            "size": size,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            },
            "_source": True
        }

        response = self.client.search(
            index=self.config.index_name,
            body=bool_query
        )

        hits = []
        for hit in response["hits"]["hits"]:
            if hit["_score"] >= min_score:
                hits.append(hit)

        return hits

    def _merge_results(
        self,
        semantic_hits: List[Dict[str, Any]],
        keyword_hits: List[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Merge and rank results from semantic and keyword search.

        Args:
            semantic_hits: Results from semantic search
            keyword_hits: Results from keyword search

        Returns:
            Combined and ranked search results
        """
        # Create score lookup by chunk_id
        semantic_scores = {
            hit["_id"]: hit["_score"] for hit in semantic_hits
        }
        keyword_scores = {
            hit["_id"]: hit["_score"] for hit in keyword_hits
        }

        # Normalize scores to [0,1] range
        if semantic_scores:
            max_semantic = max(semantic_scores.values())
            semantic_scores = {
                k: v/max_semantic for k, v in semantic_scores.items()
            }
        if keyword_scores:
            max_keyword = max(keyword_scores.values())
            keyword_scores = {
                k: v/max_keyword for k, v in keyword_scores.items()
            }

        # Combine all unique hits
        all_hits = {}
        for hit in semantic_hits + keyword_hits:
            if hit["_id"] not in all_hits:
                all_hits[hit["_id"]] = hit

        # Calculate combined scores
        results = []
        for chunk_id, hit in all_hits.items():
            semantic_score = semantic_scores.get(chunk_id, 0)
            keyword_score = keyword_scores.get(chunk_id, 0)

            combined_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score
            )

            source = hit["_source"]
            results.append(SearchResult(
                text=source["text"],
                score=combined_score,
                metadata=source["metadata"],
                source_type=source["source_type"],
                profile_id=source["profile_id"],
                chunk_id=chunk_id
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit to max results
        return results[:self.config.max_results]

    def _optimize_context_window(
        self,
        results: List[SearchResult]
    ) -> Optional[str]:
        """Optimize context window for response generation.

        Args:
            results: Retrieved search results

        Returns:
            Optimized context string
        """
        if not results:
            return None

        # Group results by profile_id
        profile_chunks = {}
        for result in results:
            if result.profile_id not in profile_chunks:
                profile_chunks[result.profile_id] = []
            profile_chunks[result.profile_id].append(result)

        # Sort chunks within each profile by score
        for chunks in profile_chunks.values():
            chunks.sort(key=lambda x: x.score, reverse=True)

        # Build context window, prioritizing high-scoring chunks
        context_parts = []
        total_length = 0

        for chunks in profile_chunks.values():
            for chunk in chunks:
                if total_length + len(chunk.text) <= self.config.context_window_size:
                    context_parts.append(chunk.text)
                    total_length += len(chunk.text)
                else:
                    break
            if total_length >= self.config.context_window_size:
                break

        return "\n\n".join(context_parts) if context_parts else None

    def retrieve(
        self,
        question: ProcessedQuestion,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a processed question.

        Args:
            question: Processed question with entities
            strategy: Retrieval strategy to use

        Returns:
            Retrieval results
        """
        start_time = time.time()

        # Get query embedding
        query_embedding = self.embeddings.embed_query(
            question.formulated_query or question.normalized_text
        )

        if strategy == RetrievalStrategy.SEMANTIC:
            semantic_hits = self._semantic_search(
                query_embedding,
                size=self.config.max_results,
                min_score=self.config.min_score
            )
            keyword_hits = []

        elif strategy == RetrievalStrategy.KEYWORD:
            semantic_hits = []
            keyword_hits = self._keyword_search(
                question.formulated_query or question.normalized_text,
                question.entities,
                size=self.config.max_results,
                min_score=self.config.min_score
            )

        else:  # HYBRID
            semantic_hits = self._semantic_search(
                query_embedding,
                size=self.config.max_results * 2,  # Get more for better hybrid ranking
                min_score=self.config.min_score
            )
            keyword_hits = self._keyword_search(
                question.formulated_query or question.normalized_text,
                question.entities,
                size=self.config.max_results * 2,
                min_score=self.config.min_score
            )

        # Merge and rank results
        results = self._merge_results(semantic_hits, keyword_hits)

        # Optimize context window
        context_window = self._optimize_context_window(results)

        execution_time = time.time() - start_time

        return RetrievalResult(
            results=results,
            strategy_used=strategy,
            total_found=len(results),
            execution_time=execution_time,
            query_entities=question.entities,
            context_window=context_window
        )
