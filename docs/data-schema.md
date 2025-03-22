# Data Schema Documentation

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

## Notes
- All text fields support Unicode
- Arrays can be empty but not null
- Nested objects must include all required fields
