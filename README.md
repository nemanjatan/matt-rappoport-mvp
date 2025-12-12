# Installment Agreement Data Extractor

A proof-of-concept application for extracting structured data from installment credit agreement images using Google Cloud Vision API for OCR and OpenAI for post-processing.

## Features

- **Google Cloud Vision OCR** - High-quality text extraction from images
- **Rule-based Extraction** - Deterministic field extraction using keyword proximity
- **OpenAI Enhancement** - Optional AI-powered improvement when OCR confidence is low
- **Streamlit UI** - Simple web interface for testing and visualization
- **Structured Output** - Canonical schema with normalized values

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Google Cloud credentials:**
   - Place your service account JSON file in the project root
   - Default: `matt-481014-e5ff3d867b2a.json`
   - Or set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

3. **Set up OpenAI (required):**
   - Create a `.env` file in the project root
   - Add: `OPENAI_API_KEY=your-openai-api-key-here`
   - OpenAI is always enabled for enhanced accuracy and validation

## Usage

### Streamlit UI (Recommended)

Launch the web interface:

```bash
streamlit run app.py
```

The UI provides:
- Image upload (drag-and-drop or file picker)
- Quick test buttons for sample images
- Toggle for OpenAI assistance
- Structured results display (fields, JSON, raw OCR)
- Download results as JSON

### Command Line

Use the pipeline directly:

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    credentials_path="matt-481014-e5ff3d867b2a.json",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = pipeline.extract("examples/IMG_1805.png")
print(result.schema.buyer_name)
print(result.schema.amount_financed)
```

### Test Scripts

Test the full pipeline:

```bash
python test_pipeline.py
```

Test individual components:

```bash
python test_ocr.py          # Test OCR extraction
python test_schema.py       # Test schema validation
python test_extraction.py   # Test deterministic extraction
python test_openai_extraction.py  # Test OpenAI enhancement
```

## Project Structure

```
src/
  ocr/          # Google Cloud Vision API integration
  processors/   # Text processing (OpenAI integration)
  extractors/   # Field extraction logic
  validators/   # Data validation
  pipeline/     # Main orchestration
  schema/       # Canonical data schema
app.py          # Streamlit UI
requirements.txt
```

## Sample Images

The project includes two test images:
- `examples/IMG_1805.png` - Home solicitation installment agreement
- `examples/IMG_1807.png` - Retail installment contract

## Extracted Fields

The pipeline extracts 13 fields:

**Buyer Information:**
- Buyer's name
- Co-buyer's name
- Street address
- Phone number

**Purchase Details:**
- Quantity
- Items purchased
- Make or model

**Financial Data:**
- Amount financed
- Finance charge
- APR
- Total of payments

**Payment Details:**
- Number of payments
- Amount of payments

## Output Format

Results are returned as `InstallmentAgreementSchema` with:
- Normalized currency values (Decimal)
- Formatted phone numbers (XXX-XXX-XXXX)
- Normalized percentages (Decimal)
- Proper null handling for missing fields

## Configuration

### Confidence Thresholds

OpenAI is automatically used when:
- Word-level mean confidence < 0.85
- Word-level min confidence < 0.80
- Block-level mean confidence < 0.85
- More than 20% of words have low confidence

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS` - Path to Google Cloud credentials
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `EXTRACTION_DEBUG` - Set to `true` to enable debug mode logging

## Debugging

### Enable Debug Mode

Set the `EXTRACTION_DEBUG` environment variable:

```bash
export EXTRACTION_DEBUG=true
python test_pipeline.py
```

Or use the test script:

```bash
python test_debug_mode.py
```

### What Gets Logged

**Normal Mode:**
- Extraction results
- OpenAI usage
- Processing time
- Confidence scores

**Debug Mode:**
- All of the above, plus:
- Raw OCR text
- All extraction candidates
- OpenAI prompts and responses (redacted)
- Detailed extraction trace

### Tracing Extraction Failures

With debug mode enabled, you can trace why a field was or wasn't extracted:

1. **Check OCR text** - See if the label/value appears in raw OCR
2. **Review candidates** - See what candidates were found for each field
3. **Check resolution** - See why a candidate was selected or rejected
4. **Review OpenAI** - See if OpenAI was used and what it returned

See `src/utils/README.md` for detailed debugging documentation.

## License

This is a proof-of-concept project for demonstration purposes.

