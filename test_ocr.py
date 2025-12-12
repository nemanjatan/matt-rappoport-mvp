"""Test script for OCR extraction on example images."""

import json
from pathlib import Path
from src.ocr import VisionOCRClient

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Example images to test
EXAMPLE_IMAGES = [
    "examples/IMG_1805.png",
    "examples/IMG_1807.png"
]


def test_ocr_extraction():
    """Test OCR extraction on example images."""
    # Initialize OCR client
    client = VisionOCRClient(credentials_path=CREDENTIALS_PATH)
    
    results = {}
    
    for image_path in EXAMPLE_IMAGES:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        if not Path(image_path).exists():
            print(f"ERROR: Image not found: {image_path}")
            continue
        
        try:
            # Extract text
            result = client.extract_text(
                image_path=image_path,
                save_raw_output=True,
                output_dir="output"
            )
            
            # Display results
            print(f"\n✓ OCR extraction successful!")
            print(f"  Full text length: {len(result.full_text)} characters")
            print(f"  Number of words: {len(result.word_annotations)}")
            print(f"  Number of blocks: {len(result.block_annotations)}")
            
            # Display confidence scores
            if result.confidence_scores:
                print(f"\n  Confidence Scores:")
                if 'word_level' in result.confidence_scores:
                    wl = result.confidence_scores['word_level']
                    print(f"    Word-level - Mean: {wl['mean']:.3f}, Min: {wl['min']:.3f}, Max: {wl['max']:.3f}")
                if 'block_level' in result.confidence_scores:
                    bl = result.confidence_scores['block_level']
                    print(f"    Block-level - Mean: {bl['mean']:.3f}, Min: {bl['min']:.3f}, Max: {bl['max']:.3f}")
            
            # Display warnings
            if result.warnings:
                print(f"\n  Warnings: {result.warnings}")
            else:
                print(f"\n  No warnings detected")
            
            # Display first 500 characters of extracted text
            print(f"\n  Extracted text preview (first 500 chars):")
            print(f"  {result.full_text[:500]}...")
            
            # Store result
            results[image_path] = {
                'full_text_length': len(result.full_text),
                'word_count': len(result.word_annotations),
                'block_count': len(result.block_annotations),
                'confidence_scores': result.confidence_scores,
                'warnings': result.warnings,
                'preview': result.full_text[:200]
            }
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results[image_path] = {'error': str(e)}
    
    # Save summary
    summary_path = Path("output/ocr_test_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Test summary saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    test_ocr_extraction()

