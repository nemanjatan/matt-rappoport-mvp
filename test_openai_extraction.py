"""Test script for OpenAI-enhanced field extraction."""

import os
import json
from src.ocr import VisionOCRClient
from src.extractors import EnhancedExtractor
from src.processors import OpenAIProcessor
from src.schema import InstallmentAgreementSchema

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test image
TEST_IMAGE = "examples/IMG_1805.png"


def test_openai_extraction():
    """Test OpenAI-enhanced extraction on IMG_1805.png."""
    print("=" * 60)
    print("Testing OpenAI-Enhanced Field Extraction")
    print("=" * 60)
    print(f"\nImage: {TEST_IMAGE}\n")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Perform OCR
    print("Step 1: Performing OCR...")
    client = VisionOCRClient(credentials_path=CREDENTIALS_PATH)
    ocr_result = client.extract_text(
        image_path=TEST_IMAGE,
        save_raw_output=False
    )
    print(f"✓ OCR complete: {len(ocr_result.full_text)} characters, {len(ocr_result.word_annotations)} words")
    
    # Display confidence scores
    if ocr_result.confidence_scores:
        word_level = ocr_result.confidence_scores.get('word_level', {})
        print(f"  Word-level confidence: mean={word_level.get('mean', 0):.3f}, min={word_level.get('min', 0):.3f}")
    
    # Step 2: Initialize OpenAI processor
    print("\nStep 2: Initializing OpenAI processor...")
    openai_processor = OpenAIProcessor(api_key=openai_key)
    
    # Check if OpenAI should be used
    should_use = openai_processor.should_use_openai(ocr_result)
    print(f"  Should use OpenAI: {should_use}")
    
    # Step 3: Extract fields with OpenAI enhancement
    print("\nStep 3: Extracting fields (with OpenAI enhancement)...")
    extractor = EnhancedExtractor(
        ocr_result=ocr_result,
        openai_processor=openai_processor,
        force_openai=True  # Force OpenAI for testing
    )
    schema = extractor.extract_all_fields()
    
    # Step 4: Display results
    print("\n" + "=" * 60)
    print("Extracted Fields:")
    print("=" * 60)
    
    results = {}
    for field_name in InstallmentAgreementSchema.model_fields.keys():
        value = getattr(schema, field_name)
        results[field_name] = value
        status = "✓" if value is not None else "✗"
        print(f"{status} {field_name:25s}: {value}")
    
    # Step 5: Show JSON output
    print("\n" + "=" * 60)
    print("JSON Output:")
    print("=" * 60)
    json_output = schema.to_json_dict()
    print(json.dumps(json_output, indent=2, ensure_ascii=False))
    
    # Step 6: Compare with expected output
    print("\n" + "=" * 60)
    print("Expected Output (from ticket):")
    print("=" * 60)
    expected = {
        "buyer": {
            "name": "Hannah Hornberger",
            "address": "500 Ricky Street, Sene, CA 71760",
            "phone": "717-257-0626"
        },
        "co_buyer": {
            "name": "Randy Hornberger",
            "address": "500 Ricky Street, Sene, CA 71760",
            "phone": "717-603-2240"
        },
        "purchase": {
            "quantity": 2,
            "items": "Appliances",
            "make_model": "Platinum Couture / Cutler"
        },
        "financial": {
            "apr": "21%",
            "finance_charge": "3,607.72",
            "amount_financed": "6,998.00",
            "total_of_payments": "11,025.76"
        },
        "payment": {
            "number_of_payments": 48,
            "payment_amount": "229.70"
        }
    }
    print(json.dumps(expected, indent=2, ensure_ascii=False))
    
    # Step 7: Save results
    output_path = "output/extraction_results_IMG_1805_openai.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Extraction test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_openai_extraction()

