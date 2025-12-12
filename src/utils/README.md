# Logging and Debugging Utilities

Comprehensive logging system for transparent debugging of extraction failures.

## Features

- **OCR Logging**: Raw text, confidence scores, word/block counts
- **Extraction Logging**: Candidate values, resolution process
- **OpenAI Logging**: Prompts and responses (with redaction)
- **Pipeline Logging**: Full extraction trace
- **Debug Mode**: Detailed logging for developers

## Usage

### Enable Debug Mode

Set environment variable:
```bash
export EXTRACTION_DEBUG=true
```

Or in Python:
```python
from src.utils import setup_logger
logger = setup_logger(debug_mode=True)
```

### Basic Logging

Logging is automatically enabled when using the pipeline:

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(credentials_path="credentials.json")
result = pipeline.extract("image.png")
# Logs are automatically printed
```

### What Gets Logged

#### OCR Extraction
- Text length and word/block counts
- Confidence scores (word-level and block-level)
- Warnings (if any)
- Raw OCR text (in debug mode)

#### Deterministic Extraction
- Field candidates found for each field
- Distance and confidence for each candidate
- Final extracted value
- Source (deterministic/openai)

#### OpenAI Enhancement
- Whether OpenAI was used and why
- Request prompt (redacted)
- Response (redacted)
- Field improvements (before/after)

#### Pipeline Flow
- Processing steps
- Processing time
- OpenAI usage decision
- Final results summary

## Debug Mode vs Normal Mode

### Normal Mode (Default)
- Logs INFO level messages
- Shows extraction results
- Shows OpenAI usage
- Minimal output

### Debug Mode (`EXTRACTION_DEBUG=true`)
- Logs DEBUG level messages
- Shows raw OCR text
- Shows all extraction candidates
- Shows OpenAI prompts and responses
- Detailed extraction trace

## Log Format

### Normal Mode
```
INFO - OCR EXTRACTION COMPLETE
INFO - Text length: 2430 characters
INFO - Words extracted: 507
INFO - Word-level confidence: mean=0.949, min=0.348, max=0.995
```

### Debug Mode
```
2024-01-15 10:30:45 - extraction_pipeline - DEBUG - RAW OCR TEXT:
2024-01-15 10:30:45 - extraction_pipeline - DEBUG - INSTALLMENT CREDIT AGREEMENT...
2024-01-15 10:30:45 - extraction_pipeline - DEBUG -   Field: buyer_name
2024-01-15 10:30:45 - extraction_pipeline - DEBUG -     Found 3 candidate(s):
2024-01-15 10:30:45 - extraction_pipeline - DEBUG -       1. 'Hannah Hornberger' (distance: 45.2, confidence: 0.987)
```

## Sensitive Data Redaction

OpenAI prompts and responses are automatically redacted:
- API keys (sk-... patterns)
- Google Cloud credentials
- Other sensitive patterns

To disable redaction (not recommended):
```python
from src.utils import log_openai_request
log_openai_request(logger, prompt, model, redact=False)
```

## Tracing Extraction Failures

### Example: Field Not Extracted

With debug mode, you can see:
1. **OCR Text**: Check if the label/value is in the OCR text
2. **Candidates**: See if any candidates were found
3. **Resolution**: See why a candidate was/wasn't selected
4. **OpenAI**: See if OpenAI was used and what it returned

### Example Debug Output

```
============================================================
DETERMINISTIC EXTRACTION
============================================================
  Field: buyer_name
    Found 3 candidate(s):
      1. 'Hannah Hornberger' (distance: 45.2, confidence: 0.987)
      2. 'Hannah' (distance: 120.5, confidence: 0.923)
      3. 'Hornberger' (distance: 200.1, confidence: 0.901)
  ✓ buyer_name               : Hannah Hornberger (deterministic)
  Field: phone_number
    No candidates found
  ✗ phone_number             : None (deterministic)
============================================================
```

This shows:
- `buyer_name` was found with 3 candidates, best one selected
- `phone_number` had no candidates, so returned None

## Integration

Logging is integrated into:
- `VisionOCRClient` - OCR extraction
- `DeterministicExtractor` - Field extraction
- `EnhancedExtractor` - OpenAI integration
- `ExtractionPipeline` - Full pipeline

All components automatically use logging when available.

## Customization

### Change Log Level
```python
from src.utils import setup_logger
logger = setup_logger(level=logging.WARNING)  # Only warnings and errors
```

### Custom Logger
```python
import logging
from src.utils import get_logger

logger = get_logger()
logger.addHandler(custom_handler)
```

## Best Practices

1. **Enable debug mode** when investigating extraction failures
2. **Check OCR text** first - if label/value isn't there, extraction will fail
3. **Review candidates** - see what the extractor found
4. **Check OpenAI logs** - see what OpenAI was asked and what it returned
5. **Review confidence scores** - low confidence may indicate OCR issues

