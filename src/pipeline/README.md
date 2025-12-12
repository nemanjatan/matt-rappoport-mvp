# Extraction Pipeline

Full end-to-end pipeline for extracting structured data from installment credit agreement images.

## Pipeline Flow

The pipeline orchestrates the following steps:

1. **Image Upload** - Validates image format and existence
2. **Google Cloud Vision OCR** - Extracts text using DOCUMENT_TEXT_DETECTION
3. **Rule-based Extraction** - Uses deterministic extractor with keyword proximity
4. **Confidence Evaluation** - Checks OCR confidence scores
5. **Optional OpenAI Post-processing** - Enhances extraction when confidence is low
6. **Final Structured Output** - Returns normalized schema with metadata

## Usage

### Basic Usage

```python
from src.pipeline import ExtractionPipeline

# Initialize pipeline
pipeline = ExtractionPipeline(
    credentials_path="matt-481014-e5ff3d867b2a.json",
    openai_api_key="your-key-here"  # Optional
)

# Extract from image file
result = pipeline.extract("examples/IMG_1805.png")

# Access extracted data
print(result.schema.buyer_name)
print(result.schema.amount_financed)

# Get full results
output_dict = result.to_dict()
```

### With OpenAI (Automatic)

OpenAI is automatically used when:
- OCR confidence is below threshold (mean < 0.85 or min < 0.80)
- More than 20% of words have low confidence
- OCR warnings are present

```python
# OpenAI will be used automatically if confidence is low
pipeline = ExtractionPipeline(
    credentials_path="credentials.json",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = pipeline.extract("image.png")
print(f"Used OpenAI: {result.used_openai}")
```

### Force OpenAI

Force OpenAI usage regardless of confidence:

```python
pipeline = ExtractionPipeline(
    credentials_path="credentials.json",
    openai_api_key="your-key",
    force_openai=True  # Always use OpenAI
)
```

### Extract from Bytes

For use with uploaded files (e.g., Streamlit):

```python
with open("image.png", "rb") as f:
    image_bytes = f.read()

result = pipeline.extract_from_bytes(image_bytes, image_format="PNG")
```

## Output Structure

The `ExtractionResult` contains:

- `schema`: `InstallmentAgreementSchema` with all extracted fields
- `ocr_result`: Full OCR result from Google Cloud Vision
- `used_openai`: Boolean indicating if OpenAI was used
- `confidence_scores`: OCR confidence metrics
- `processing_time`: Total processing time in seconds

## Example Output

```python
result = pipeline.extract("examples/IMG_1805.png")

# Access schema fields
result.schema.buyer_name          # "Hannah Hornberger"
result.schema.phone_number         # "717-257-0626"
result.schema.amount_financed     # Decimal("6998.00")
result.schema.apr                 # Decimal("21")

# Get metadata
result.used_openai                 # True/False
result.confidence_scores           # {'word_level': {...}, 'block_level': {...}}
result.processing_time             # 1.23 (seconds)

# Convert to dictionary
output = result.to_dict()
# {
#   "extracted_data": {...},
#   "metadata": {
#     "used_openai": true,
#     "confidence_scores": {...},
#     "processing_time_seconds": 1.23,
#     ...
#   }
# }
```

## Error Handling

The pipeline handles errors gracefully:

- **Invalid image format**: Raises `ValueError`
- **Missing image file**: Raises `FileNotFoundError`
- **OCR API failure**: Raises exception from Google Cloud Vision
- **OpenAI failure**: Falls back to deterministic extraction
- **Schema validation failure**: Returns initial extraction with warnings

## Testing

Run the full pipeline test:

```bash
python test_pipeline.py
```

This tests both sample images and compares results with expected outputs.

## Configuration

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials (if not passed directly)
- `OPENAI_API_KEY`: OpenAI API key (if not passed directly)

### Confidence Thresholds

Adjust in `OpenAIProcessor`:

```python
from src.processors import OpenAIProcessor

processor = OpenAIProcessor()
processor.LOW_CONFIDENCE_THRESHOLD = 0.90  # More aggressive
processor.LOW_WORD_CONFIDENCE_THRESHOLD = 0.75  # Less aggressive
```

## Pipeline Components

1. **VisionOCRClient** (`src.ocr`) - Google Cloud Vision API integration
2. **DeterministicExtractor** (`src.extractors`) - Rule-based field extraction
3. **EnhancedExtractor** (`src.extractors`) - Combines deterministic + OpenAI
4. **OpenAIProcessor** (`src.processors`) - OpenAI enhancement when needed
5. **InstallmentAgreementSchema** (`src.schema`) - Canonical data schema

## Performance

Typical processing times:
- OCR only: ~0.5-1.0 seconds
- OCR + Deterministic extraction: ~1.0-1.5 seconds
- OCR + Deterministic + OpenAI: ~2.0-4.0 seconds

Processing time depends on:
- Image size and complexity
- OCR text length
- Whether OpenAI is used

