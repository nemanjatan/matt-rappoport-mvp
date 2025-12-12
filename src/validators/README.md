# AI-Powered Validation and Correction

Intelligent post-processing layer that uses AI to normalize, validate, and correct extracted data without hardcoding complex rules.

## Overview

The `AIValidator` class provides:
- **Name Normalization**: Ensures buyer and co-buyer share consistent last names
- **Address Correction**: Fixes incomplete or incorrect addresses
- **OCR Error Correction**: Fixes common OCR character confusions
- **Relationship Validation**: Ensures buyer/co-buyer share addresses when appropriate
- **Validation Signal Detection**: Automatically detects issues and triggers corrections

## Key Features

### 1. Name Normalization

**Problem**: OCR may extract "Hannah Hornberse" and "Randy Hurnberge" as different last names.

**Solution**: AI detects last name mismatch and normalizes to consistent spelling:
- Compares last names using similarity algorithm
- Uses OCR context to determine correct spelling
- Normalizes both names to match

### 2. Address Correction

**Problem**: Addresses may be incomplete ("Suu" instead of "500 Ricky Street") or contain phone numbers.

**Solution**: AI corrects addresses by:
- Detecting incomplete addresses (too short, missing street number)
- Removing phone numbers incorrectly included
- Using OCR context to find complete address
- Ensuring buyer and co-buyer share same address when appropriate

### 3. Validation Signal Detection

The validator automatically detects:
- **Last name mismatches** (high severity)
- **Address too short** (high severity)
- **Address contains phone** (high severity)
- **Missing street number** (medium severity)
- **Possible OCR errors** (medium severity)

Only issues with medium or high severity trigger AI correction.

## Usage

The validator is automatically integrated into the pipeline:

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    credentials_path="credentials.json",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = pipeline.extract("image.png")

# Validation is automatically applied
# Check validation results
if result.validation_result:
    print(f"Issues found: {len(result.validation_result.issues_found)}")
    print(f"Corrections applied: {len(result.validation_result.corrections_applied)}")
```

## How It Works

### Step 1: Issue Detection

The validator analyzes the extracted schema and detects issues:
- Compares buyer/co-buyer last names
- Checks address quality and completeness
- Detects OCR-like errors in names
- Identifies validation signals

### Step 2: AI Correction (if needed)

If medium or high severity issues are found:
1. Builds correction prompt with:
   - Detected issues
   - Current extracted data
   - OCR text for context
2. Sends to OpenAI with specific instructions:
   - Normalize names (especially last names)
   - Correct addresses
   - Ensure buyer/co-buyer consistency
   - Only correct what's clearly visible in OCR
3. Returns corrected schema

### Step 3: Result

Returns `ValidationResult` with:
- Corrected schema
- Issues found
- Corrections applied
- Whether AI was used

## Validation Rules

### Name Consistency

- If buyer and co-buyer appear to be a couple (same address), ensure they share the same last name
- Fix OCR errors in names (e.g., "Hornberse" → "Hornberger")
- Use OCR context to determine correct spelling

### Address Correction

- Addresses must start with a street number
- Remove phone numbers from addresses
- If buyer and co-buyer share an address, ensure both have the same address
- Use OCR context to find complete address

### OCR Error Correction

Common fixes:
- "Hornberse" → "Hornberger"
- "Hurnberge" → "Hornberger"
- "Suu" → "500 Ricky Street" (using OCR context)

## Example

### Before Validation:
```json
{
  "buyer_name": "Hannah Hornberse",
  "co_buyer_name": "Randy Hurnberge",
  "street_address": "Suu"
}
```

### After Validation:
```json
{
  "buyer_name": "Hannah Hornberger",
  "co_buyer_name": "Randy Hornberger",
  "street_address": "500 Ricky Street, Sene, CA 71760"
}
```

## Integration

The validator is automatically used when:
- OpenAI API key is available
- Issues are detected in extracted data
- Issues have medium or high severity

No manual configuration needed - it's part of the pipeline.

## Logging

Validation is logged when debug mode is enabled:

```
INFO - Step 3: AI validation and correction...
INFO -   Applied 3 correction(s)
INFO -     - Corrected buyer_name/co_buyer_name: Last names don't match
INFO -     - Corrected street_address: Address seems incomplete
INFO -     - Corrected street_address: Address missing street number
INFO -   Found 3 issue(s)
```

## Best Practices

1. **Always use OpenAI API key** - Validation only works with OpenAI
2. **Enable debug mode** - See what issues were detected and corrected
3. **Review validation results** - Check `result.validation_result` for details
4. **Trust the AI** - It only corrects what's clearly visible in OCR text

## Limitations

- Requires OpenAI API key
- Only corrects issues that are clearly visible in OCR text
- May not catch all edge cases
- Adds processing time (~1-2 seconds)

## Future Enhancements

Potential improvements:
- Support for Gemini API as alternative
- More sophisticated name matching algorithms
- Address validation against postal databases
- Confidence scoring for corrections

