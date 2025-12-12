"""Test script to demonstrate debug mode logging."""

import os
from src.pipeline import ExtractionPipeline

# Enable debug mode
os.environ["EXTRACTION_DEBUG"] = "true"

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test image
TEST_IMAGE = "examples/IMG_1807.png"


def test_debug_mode():
    """Test extraction with debug mode enabled."""
    print("=" * 60)
    print("DEBUG MODE TEST")
    print("=" * 60)
    print(f"Debug mode: {os.getenv('EXTRACTION_DEBUG', 'false')}")
    print(f"Image: {TEST_IMAGE}\n")
    
    # Initialize pipeline
    openai_key = os.getenv("OPENAI_API_KEY")
    pipeline = ExtractionPipeline(
        credentials_path=CREDENTIALS_PATH,
        openai_api_key=openai_key,
        force_openai=False
    )
    
    # Run extraction (logs will be printed)
    result = pipeline.extract(TEST_IMAGE, save_raw_ocr=False)
    
    print("\n" + "=" * 60)
    print("EXTRACTION RESULT SUMMARY")
    print("=" * 60)
    print(f"Buyer name: {result.schema.buyer_name}")
    print(f"Amount financed: {result.schema.amount_financed}")
    print(f"Used OpenAI: {result.used_openai}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    test_debug_mode()

