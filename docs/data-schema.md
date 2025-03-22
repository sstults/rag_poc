# Data Schema Documentation

This document outlines the required format for HR profile data that will be processed by the RAG pipeline.

## Overview

The HR data pipeline processes JSON documents containing professional profiles. Each profile is processed into semantic chunks, embedded using Amazon Bedrock, and indexed in OpenSearch for efficient retrieval.

## HR Profile Schema

```json
{
  "profile": {
    "id": "string",
    "basic_info": {
      "name": "string",
      "location": "string",
      "current_role": "string"
    },
    "experience": [{
      "company": "string",
      "role": "string",
      "duration": "string",
      "responsibilities": ["string"]
    }],
    "skills": [{
      "category": "string",
      "skills": ["string"]
    }],
    "qa_data": [{
      "question": "string",
      "answer": "string",
      "context": "string"
    }]
  }
}
```

## Field Requirements

### Basic Info (Required)
- `id`: Unique identifier for the profile (UUID v4 format). If not provided, one will be generated.
- `basic_info`: Core information about the candidate
  - `name`: Full name of the candidate
  - `location`: Geographic location (city/state/country)
  - `current_role`: Current job title or role

### Experience (Required)
Array of professional experiences, each containing:
- `company`: Company or organization name
- `role`: Job title or position held
- `duration`: Time period in the role (e.g., "2020-2023" or "3 years")
- `responsibilities`: Array of key responsibilities or achievements
  - Each responsibility should be a clear, concise statement
  - Recommended length: 1-3 sentences per responsibility
  - Include measurable outcomes where possible

### Skills (Required)
Array of skill sets, each containing:
- `category`: Skill category or domain (e.g., "Programming Languages", "Cloud Technologies")
- `skills`: Array of specific skills within the category
  - Each skill should be a specific technology, tool, or capability
  - Use standard industry terminology where possible

### Q&A Data (Optional)
Array of interview questions and answers, each containing:
- `question`: Interview question or prompt
- `answer`: Candidate's response
- `context`: Additional context or notes about the response
  - Can include interviewer observations
  - May contain relevant background information

## Processing Behavior

1. **Chunking**
   - Text is split into semantic chunks of approximately 1000 characters
   - Chunk boundaries respect natural content breaks
   - Each chunk maintains contextual information from the profile

2. **Metadata**
   - Each chunk is enriched with metadata including:
     - Profile ID and chunk ID
     - Source type (overview, experience, skills, qa)
     - Basic profile information
     - Relationships to adjacent chunks

3. **Embeddings**
   - Each chunk is embedded using Amazon Bedrock
   - Embedding dimension: 1536
   - Embeddings are optimized for semantic similarity search

4. **Indexing**
   - Chunks are indexed in OpenSearch with:
     - kNN vector search capability for embeddings
     - Full-text search on content
     - Filterable metadata fields

## Data Quality Guidelines

1. **Text Content**
   - All text fields support Unicode characters
   - Avoid special formatting (markdown, HTML)
   - Use proper capitalization and punctuation
   - Keep content professional and factual

2. **Arrays**
   - Can be empty but not null
   - Recommended maximum items:
     - Responsibilities: 10 per experience
     - Skills: 15 per category
     - Q&A entries: 20 per profile

3. **Field Length Guidelines**
   - Company/role names: 100 characters max
   - Responsibility statements: 500 characters max
   - Q&A answers: 2000 characters max
   - Skills: 50 characters max per skill

## Example Profile

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "basic_info": {
    "name": "Jane Smith",
    "location": "Seattle, WA, USA",
    "current_role": "Senior Software Engineer"
  },
  "experience": [
    {
      "company": "Tech Corp",
      "role": "Software Engineer",
      "duration": "2020-2023",
      "responsibilities": [
        "Led development of microservices architecture serving 1M+ daily users",
        "Reduced API response time by 40% through optimization of database queries",
        "Mentored 5 junior developers in best practices and architecture patterns"
      ]
    }
  ],
  "skills": [
    {
      "category": "Programming Languages",
      "skills": ["Python", "Java", "TypeScript"]
    },
    {
      "category": "Cloud Technologies",
      "skills": ["AWS Lambda", "Docker", "Kubernetes"]
    }
  ],
  "qa_data": [
    {
      "question": "Describe a challenging technical problem you solved recently",
      "answer": "I encountered a performance bottleneck in our payment processing system...",
      "context": "Demonstrated strong problem-solving skills and system optimization knowledge"
    }
  ]
}
```

## Error Handling

1. **Missing Required Fields**
   - Processing will fail if required fields are missing
   - Error message will indicate missing fields

2. **Invalid Data Types**
   - String expected where number provided
   - Array expected where object provided
   - Error message will indicate type mismatch

3. **Malformed JSON**
   - Processing will fail with syntax error details
   - Validation errors will be logged

## Best Practices

1. **Profile Organization**
   - List experiences in reverse chronological order
   - Group skills by logical categories
   - Ensure Q&A responses are comprehensive but concise

2. **Content Quality**
   - Use specific, quantifiable achievements
   - Maintain consistent terminology
   - Provide context where necessary

3. **Updates and Maintenance**
   - Use consistent ID for profile updates
   - Maintain version history if needed
   - Regular validation of data format
