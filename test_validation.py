"""Test script for AI validation and correction."""

import os
import json
from src.pipeline import ExtractionPipeline

# Enable debug mode to see validation details
os.environ["EXTRACTION_DEBUG"] = "true"

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test image with known issues
TEST_IMAGE = "examples/IMG_1805.png"


def test_validation():
    """Test AI validation and correction."""
    print("=" * 60)
    print("AI VALIDATION AND CORRECTION TEST")
    print("=" * 60)
    print(f"Image: {TEST_IMAGE}\n")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize pipeline (will automatically use AI validator)
    pipeline = ExtractionPipeline(
        credentials_path=CREDENTIALS_PATH,
        openai_api_key=openai_key,
        force_openai=False
    )
    
    print("Running extraction with AI validation...\n")
    
    # Run extraction
    result = pipeline.extract(TEST_IMAGE, save_raw_ocr=False)
    
    # Display results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    
    schema = result.schema
    print(f"\nBuyer Information:")
    print(f"  Buyer name: {schema.buyer_name}")
    print(f"  Co-buyer name: {schema.co_buyer_name}")
    print(f"  Street address: {schema.street_address}")
    print(f"  Phone number: {schema.phone_number}")
    
    # Check validation results
    if result.validation_result:
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        validation = result.validation_result
        print(f"\nIssues found: {len(validation.issues_found)}")
        for issue in validation.issues_found:
            print(f"  - {issue.field}: {issue.issue_type} ({issue.severity})")
            print(f"    {issue.description}")
        
        print(f"\nCorrections applied: {len(validation.corrections_applied)}")
        for correction in validation.corrections_applied:
            print(f"  - {correction}")
        
        print(f"\nUsed AI: {validation.used_ai}")
    
    # Save results
    output_path = "output/validation_test_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_validation()

