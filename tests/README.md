# Test Suite

Comprehensive test suite for ensuring extraction accuracy and preventing regressions.

## Test Structure

### `test_extraction_accuracy.py`
Tests extraction accuracy for both sample images:
- **IMG_1805.png**: Tests buyer names, phone numbers, amounts, APR, payments
- **IMG_1807.png**: Tests buyer names, phone numbers, amounts, APR, payments
- Uses tolerant assertions (allows OCR variations)
- Validates key numeric values and names

### `test_schema_completeness.py`
Tests schema structure and validation:
- All fields present in extraction results
- Correct data types (string, int, Decimal)
- Schema validation and serialization
- Value normalization (phone, currency, APR)

### `test_error_handling.py`
Tests pipeline robustness:
- Empty OCR text handling
- Malformed currency values
- Missing field labels
- Low confidence OCR
- Invalid image formats
- Missing files

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_extraction_accuracy.py
```

### Run specific test:
```bash
pytest tests/test_extraction_accuracy.py::TestIMG1805Extraction::test_buyer_name
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Test Fixtures

Fixtures defined in `conftest.py`:
- `pipeline`: Basic extraction pipeline
- `pipeline_with_openai`: Pipeline with OpenAI (if available)
- `img_1805_path`: Path to IMG_1805.png
- `img_1807_path`: Path to IMG_1807.png
- `expected_img_1805`: Expected values for IMG_1805.png
- `expected_img_1807`: Expected values for IMG_1807.png

## Test Strategy

### Tolerant Assertions
Tests use tolerant assertions to account for OCR variations:
- Name matching: Checks for key name parts (e.g., "Hannah" or "Hornberger")
- Phone numbers: Compares last 10 digits (allows area code variations)
- Currency: Allows 1% tolerance for OCR parsing differences
- APR: Allows 0.5% tolerance

### Schema Validation
All tests ensure:
- Schema structure is complete
- Data types are correct
- Values are normalized properly
- Schema can be serialized/deserialized

### Error Handling
Tests verify pipeline doesn't crash on:
- Empty OCR text
- Malformed values
- Missing labels
- Low confidence OCR
- Invalid inputs

## Continuous Integration

Tests should be run:
- Before committing code
- In CI/CD pipeline
- After major refactoring
- When adding new features

## Expected Test Results

For IMG_1805.png:
- Buyer name: Contains "Hannah" or "Hornberger"
- Amount financed: ~6998.00 (within 1% tolerance)
- APR: ~21.0 (within 0.5% tolerance)
- Number of payments: 48

For IMG_1807.png:
- Buyer name: Contains "David" or "Powers"
- Amount financed: ~3644.28 (within 1% tolerance)
- APR: ~0.0 (within 0.1% tolerance)
- Number of payments: 6

## Notes

- Tests may skip if sample images are not found
- OpenAI tests require OPENAI_API_KEY environment variable
- Some tests may have lower accuracy due to OCR limitations
- Tolerant assertions account for expected OCR variations

