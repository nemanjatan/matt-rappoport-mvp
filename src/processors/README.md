# OpenAI Processor

OpenAI integration for improving extraction quality when OCR confidence is low.

## Overview

The `OpenAIProcessor` class uses OpenAI to:
- Normalize extracted values
- Resolve ambiguities in OCR text
- Validate numeric relationships
- Fill missing fields only when clearly inferable

## Key Features

### Confidence-Based Activation

OpenAI is automatically invoked when:
- Word-level mean confidence < 0.85
- Word-level min confidence < 0.80
- Block-level mean confidence < 0.85
- More than 20% of words have confidence < 0.80
- OCR warnings are present

### Strict Schema Compliance

- **No hallucination**: Only extracts values explicitly present in OCR text
- **Null for uncertainty**: Returns `null` when values are ambiguous or missing
- **Schema validation**: All outputs strictly conform to `InstallmentAgreementSchema`
- **Normalization**: Automatically normalizes currency, phone numbers, APR, etc.

## Usage

```python
from src.ocr import VisionOCRClient
from src.extractors import EnhancedExtractor
from src.processors import OpenAIProcessor

# Perform OCR
client = VisionOCRClient(credentials_path="credentials.json")
ocr_result = client.extract_text("image.png")

# Initialize OpenAI processor
openai_processor = OpenAIProcessor(api_key="your-api-key")

# Extract with OpenAI enhancement
extractor = EnhancedExtractor(
    ocr_result=ocr_result,
    openai_processor=openai_processor
)
schema = extractor.extract_all_fields()
```

## Configuration

Set the OpenAI API key via environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly:
```python
processor = OpenAIProcessor(api_key="your-api-key")
```

## Confidence Thresholds

You can adjust confidence thresholds:
```python
processor = OpenAIProcessor(api_key="your-api-key")
processor.LOW_CONFIDENCE_THRESHOLD = 0.90  # More aggressive
processor.LOW_WORD_CONFIDENCE_THRESHOLD = 0.75  # Less aggressive
```

## How It Works

1. **Deterministic Extraction First**: Always performs deterministic extraction first
2. **Confidence Check**: Evaluates OCR confidence scores
3. **OpenAI Enhancement**: If confidence is low, sends to OpenAI:
   - Raw OCR text
   - Initial extraction results
   - Candidate values from proximity search
4. **Schema Validation**: Validates OpenAI response against schema
5. **Fallback**: If OpenAI fails, returns deterministic extraction

## Prompt Engineering

The system prompt enforces:
- No hallucination or guessing
- Strict null handling for missing/ambiguous values
- Proper normalization rules
- Schema compliance
- Numeric relationship validation (when both values are visible)

## Error Handling

- If OpenAI API fails, falls back to deterministic extraction
- If OpenAI returns invalid JSON, falls back to deterministic extraction
- If schema validation fails, logs warning and uses initial schema

## Testing

Run the test script:
```bash
python test_openai_extraction.py
```

Or test confidence checking:
```bash
python test_openai_integration.py
```

