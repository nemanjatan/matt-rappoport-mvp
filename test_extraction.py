"""Test script for deterministic field extraction."""

import json
from src.ocr import VisionOCRClient
from src.extractors import DeterministicExtractor
from src.schema import InstallmentAgreementSchema

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test image
TEST_IMAGE = "examples/IMG_1807.png"


def test_extraction():
    """Test deterministic extraction on IMG_1807.png."""
    print("=" * 60)
    print("Testing Deterministic Field Extraction")
    print("=" * 60)
    print(f"\nImage: {TEST_IMAGE}\n")
    
    # Step 1: Perform OCR
    print("Step 1: Performing OCR...")
    client = VisionOCRClient(credentials_path=CREDENTIALS_PATH)
    ocr_result = client.extract_text(
        image_path=TEST_IMAGE,
        save_raw_output=False
    )
    print(f"✓ OCR complete: {len(ocr_result.full_text)} characters, {len(ocr_result.word_annotations)} words")
    
    # Step 2: Extract fields
    print("\nStep 2: Extracting fields using deterministic logic...")
    extractor = DeterministicExtractor(ocr_result)
    schema = extractor.extract_all_fields()
    
    # Step 3: Display results
    print("\n" + "=" * 60)
    print("Extracted Fields:")
    print("=" * 60)
    
    results = {}
    for field_name in InstallmentAgreementSchema.model_fields.keys():
        value = getattr(schema, field_name)
        results[field_name] = value
        status = "✓" if value is not None else "✗"
        print(f"{status} {field_name:25s}: {value}")
    
    # Step 4: Show JSON output
    print("\n" + "=" * 60)
    print("JSON Output:")
    print("=" * 60)
    json_output = schema.to_json_dict()
    print(json.dumps(json_output, indent=2, ensure_ascii=False))
    
    # Step 5: Save results
    output_path = "output/extraction_results_IMG_1807.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Step 6: Field-by-field extraction details
    print("\n" + "=" * 60)
    print("Field Extraction Details:")
    print("=" * 60)
    
    for field_name in ['buyer_name', 'phone_number', 'amount_financed', 'apr', 'number_of_payments']:
        print(f"\n{field_name}:")
        candidates = extractor._find_field_candidates(field_name)
        if candidates:
            print(f"  Found {len(candidates)} candidate(s):")
            for i, candidate in enumerate(candidates[:3], 1):  # Show top 3
                print(f"    {i}. '{candidate.value}' (distance: {candidate.distance:.1f}, confidence: {candidate.confidence:.3f})")
        else:
            print("  No candidates found")
    
    print("\n" + "=" * 60)
    print("Extraction test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_extraction()

