"""Test OpenAI integration with confidence checking."""

import os
import json
from src.ocr import VisionOCRClient
from src.extractors import EnhancedExtractor, DeterministicExtractor
from src.processors import OpenAIProcessor

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test image
TEST_IMAGE = "examples/IMG_1805.png"


def test_confidence_checking():
    """Test confidence checking logic."""
    print("=" * 60)
    print("Testing Confidence Checking")
    print("=" * 60)
    
    # Perform OCR
    client = VisionOCRClient(credentials_path=CREDENTIALS_PATH)
    ocr_result = client.extract_text(
        image_path=TEST_IMAGE,
        save_raw_output=False
    )
    
    print(f"\nOCR Confidence Scores:")
    if ocr_result.confidence_scores:
        word_level = ocr_result.confidence_scores.get('word_level', {})
        block_level = ocr_result.confidence_scores.get('block_level', {})
        print(f"  Word-level: mean={word_level.get('mean', 0):.3f}, min={word_level.get('min', 0):.3f}")
        print(f"  Block-level: mean={block_level.get('mean', 0):.3f}, min={block_level.get('min', 0):.3f}")
    
    # Check if OpenAI should be used
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        processor = OpenAIProcessor(api_key=openai_key)
        should_use = processor.should_use_openai(ocr_result)
        print(f"\nShould use OpenAI: {should_use}")
        print(f"  (Threshold: {processor.LOW_CONFIDENCE_THRESHOLD})")
    else:
        print("\nOPENAI_API_KEY not set - skipping OpenAI check")
    
    print("\n" + "=" * 60)


def test_deterministic_vs_enhanced():
    """Compare deterministic vs enhanced extraction."""
    print("\n" + "=" * 60)
    print("Comparing Deterministic vs Enhanced Extraction")
    print("=" * 60)
    
    # Perform OCR
    client = VisionOCRClient(credentials_path=CREDENTIALS_PATH)
    ocr_result = client.extract_text(
        image_path=TEST_IMAGE,
        save_raw_output=False
    )
    
    # Deterministic extraction
    print("\n1. Deterministic Extraction:")
    deterministic = DeterministicExtractor(ocr_result)
    det_schema = deterministic.extract_all_fields()
    
    det_results = {}
    for field_name in det_schema.model_fields.keys():
        value = getattr(det_schema, field_name)
        det_results[field_name] = value
        status = "✓" if value is not None else "✗"
        print(f"   {status} {field_name:25s}: {value}")
    
    # Enhanced extraction (if OpenAI available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("\n2. Enhanced Extraction (with OpenAI):")
        try:
            processor = OpenAIProcessor(api_key=openai_key)
            enhanced = EnhancedExtractor(
                ocr_result=ocr_result,
                openai_processor=processor,
                force_openai=True  # Force for testing
            )
            enh_schema = enhanced.extract_all_fields()
            
            enh_results = {}
            for field_name in enh_schema.model_fields.keys():
                value = getattr(enh_schema, field_name)
                enh_results[field_name] = value
                det_value = det_results.get(field_name)
                changed = "→" if value != det_value else " "
                status = "✓" if value is not None else "✗"
                print(f"   {status} {field_name:25s}: {value} {changed}")
            
            # Save comparison
            comparison = {
                "deterministic": det_results,
                "enhanced": enh_results
            }
            with open("output/extraction_comparison.json", 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
            print("\n✓ Comparison saved to: output/extraction_comparison.json")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("\n2. Enhanced Extraction: OPENAI_API_KEY not set")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_checking()
    test_deterministic_vs_enhanced()

