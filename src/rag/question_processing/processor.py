"""Question processing module for HR RAG system."""

import re
from typing import List, Optional, Dict, Set

from langchain_community.llms import Bedrock
from langchain_core.prompts import PromptTemplate

from .question_processing_types import (
    Entity,
    EntityType,
    ProcessedQuestion,
    QuestionContext,
    ValidationError,
)


class QuestionProcessor:
    """Processes and validates questions for the HR RAG system."""

    def __init__(self, bedrock_model_id: str = "anthropic.claude-v2"):
        """Initialize the question processor.

        Args:
            bedrock_model_id: ID of the Bedrock model to use for entity recognition
        """
        self.llm = Bedrock(model_id=bedrock_model_id)
        self.entity_patterns: Dict[EntityType, Set[str]] = self._load_entity_patterns()

        # Initialize prompt templates
        self.entity_recognition_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given this question about HR profiles, identify any entities related to:
            - Skills (e.g., Python, leadership)
            - Roles (e.g., Software Engineer, Project Manager)
            - Companies
            - Locations
            - Experience levels (e.g., senior, junior)
            - Education (e.g., Bachelor's in CS)
            - Certifications (e.g., AWS, PMP)

            Question: {question}

            Return the entities in JSON format with their types and positions:
            {{"entities": [
                {{"type": "skill", "value": "Python", "start": 25, "end": 31}},
                {{"type": "role", "value": "Software Engineer", "start": 40, "end": 56}}
            ]}}""")

    def _load_entity_patterns(self) -> Dict[EntityType, Set[str]]:
        """Load regex patterns for entity recognition."""
        return {
            EntityType.SKILL: {
                r'\b(?:Python|Java|JavaScript|TypeScript|React|Angular|Vue|Node\.js|SQL)\b',
                r'\b(?:leadership|communication|teamwork|problem[- ]solving)\b',
            },
            EntityType.ROLE: {
                r'\b(?:Software Engineer|Developer|Architect|Manager|Lead|Director)\b',
                r'\b(?:Frontend|Backend|Full[- ]Stack|DevOps|ML|Data)\b',
            },
            EntityType.EXPERIENCE_LEVEL: {
                r'\b(?:Senior|Junior|Mid[- ]Level|Principal|Staff|Entry[- ]Level)\b',
                r'\b(?:\d+[+]? years? experience)\b',
            },
            EntityType.LOCATION: {
                r'\b(?:remote|hybrid|on[- ]site|in[- ]office)\b',
                r'\b(?:[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*,\s*[A-Z]{2})\b',  # City, State
            },
            EntityType.EDUCATION: {
                r"\b(?:Bachelor'?s|Master'?s|PhD|BS|MS|BA|MA)\b",
                r'\b(?:Computer Science|CS|Engineering|Business|MBA)\b',
            },
            EntityType.CERTIFICATION: {
                r'\b(?:AWS|Azure|GCP|PMP|Scrum|CISSP|Security[+])\b',
            },
        }

    def _validate_question(self, question: str) -> List[ValidationError]:
        """Validate the input question.

        Args:
            question: The question to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Check for empty or whitespace-only questions
        if not question or not question.strip():
            errors.append(ValidationError(
                message="Question cannot be empty",
                severity="error"
            ))
            return errors

        # Check minimum length
        if len(question.strip()) < 10:
            errors.append(ValidationError(
                message="Question is too short. Please be more specific.",
                severity="warning"
            ))

        # Check maximum length
        if len(question) > 500:
            errors.append(ValidationError(
                message="Question is too long. Please be more concise.",
                severity="warning"
            ))

        # Check for common HR question patterns
        hr_patterns = [
            r'\b(?:experience|skill|background|qualification)\b',
            r'\b(?:work|project|role|position|job)\b',
            r'\b(?:education|certification|degree)\b',
            r'\b(?:team|leadership|management)\b',
        ]

        if not any(re.search(pattern, question, re.IGNORECASE) for pattern in hr_patterns):
            errors.append(ValidationError(
                message="Question may not be HR-related. Please ask about professional experience, skills, or qualifications.",
                severity="warning"
            ))

        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(?:private|confidential|sensitive)\b',
            r'\b(?:salary|compensation|pay)\b',  # May be sensitive
            r'\b(?:address|phone|email|ssn)\b',  # Personal info
            r'\b(?:race|gender|age|religion|disability)\b',  # Protected characteristics
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                errors.append(ValidationError(
                    message="Question may request sensitive or private information.",
                    severity="error"
                ))
                break

        return errors

    def _recognize_entities(self, question: str) -> List[Entity]:
        """Recognize HR-specific entities in the question.

        Args:
            question: The question to analyze

        Returns:
            List of recognized entities
        """
        entities = []

        # First use regex patterns for basic entity recognition
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, question, re.IGNORECASE):
                    entities.append(Entity(
                        type=entity_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end()
                    ))

        # Then use LLM for more sophisticated entity recognition
        llm_response = self.llm(
            self.entity_recognition_prompt.format(question=question)
        )

        try:
            # Parse LLM response and add any new entities
            # Note: In practice, would need more robust parsing and deduplication
            pass
        except Exception as e:
            # Log error but continue with regex-based entities
            print(f"Error parsing LLM response: {e}")

        return entities

    def _formulate_query(
        self,
        question: str,
        entities: List[Entity],
        context: Optional[QuestionContext] = None
    ) -> str:
        """Formulate an enhanced query for retrieval.

        Args:
            question: Original question
            entities: Recognized entities
            context: Optional question context

        Returns:
            Formulated query string
        """
        # Start with the original question
        query_parts = [question.strip()]

        # Add entity-based expansions
        for entity in entities:
            if entity.type == EntityType.SKILL:
                # Add related skills
                query_parts.append(f"technical expertise {entity.value}")
            elif entity.type == EntityType.ROLE:
                # Add role variations
                query_parts.append(f"position as {entity.value}")
                query_parts.append(f"work as {entity.value}")
            elif entity.type == EntityType.EXPERIENCE_LEVEL:
                # Add experience context
                query_parts.append(f"professional level {entity.value}")

        # Add context-based expansions
        if context and context.filters:
            for key, value in context.filters.items():
                query_parts.append(f"{key}:{value}")

        # Combine parts with weights
        # Note: In practice, would need more sophisticated query formulation
        return " ".join(query_parts)

    def process_question(
        self,
        question: str,
        context: Optional[QuestionContext] = None
    ) -> ProcessedQuestion:
        """Process and validate an input question.

        Args:
            question: The question to process
            context: Optional processing context

        Returns:
            Processed question with validation results and recognized entities
        """
        # Normalize text
        normalized = " ".join(question.split())  # Basic normalization

        # Validate
        validation_errors = self._validate_question(normalized)
        is_valid = not any(error.severity == "error" for error in validation_errors)

        # Recognize entities
        entities = self._recognize_entities(normalized) if is_valid else []

        # Formulate query if valid
        formulated_query = None
        if is_valid:
            formulated_query = self._formulate_query(normalized, entities, context)

        return ProcessedQuestion(
            original_text=question,
            normalized_text=normalized,
            entities=entities,
            validation_errors=validation_errors,
            is_valid=is_valid,
            formulated_query=formulated_query
        )
