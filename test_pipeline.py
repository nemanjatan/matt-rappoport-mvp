"""Test the full extraction pipeline on both sample images."""

import json
import os
from pathlib import Path
from src.pipeline import ExtractionPipeline

# Path to credentials file
CREDENTIALS_PATH = "matt-481014-e5ff3d867b2a.json"

# Test images
TEST_IMAGES = [
    "examples/IMG_1805.png",
    "examples/IMG_1807.png"
]

# Expected outputs (from ticket)
EXPECTED_OUTPUTS = {
    "examples/IMG_1805.png": {
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
    },
    "examples/IMG_1807.png": {
        "buyer": {
            "name": "David Powers",
            "phone": "843-333-4540",
            "address": "214 Cheyenne Trail, Liberty, SC 29657"
        },
        "co_buyer": {
            "name": "Lydia Powers",
            "phone": "843-222-1407",
            "address": "214 Cheyenne Trail, Liberty, SC 29657"
        },
        "purchase": {
            "quantity": 1,
            "items": "Food / Goods and/or Services",
            "make_model": "N/A"
        },
        "financial": {
            "amount_financed": "$3,644.28",
            "finance_charge": "$0.00",
            "apr": "0.00%",
            "total_of_payments": "$3,644.28"
        },
        "payment": {
            "number_of_payments": 6,
            "payment_amount": "$607.38 per installment"
        }
    }
}


def normalize_value(value, field_type):
    """Normalize value for comparison."""
    if value is None:
        return None
    
    if field_type == "phone":
        # Normalize phone to XXX-XXX-XXXX
        import re
        digits = re.sub(r'\D', '', str(value))
        if len(digits) == 10:
            return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        return value
    
    if field_type == "currency":
        # Remove $ and commas, compare as float
        cleaned = str(value).replace('$', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except:
            return value
    
    if field_type == "percentage":
        # Remove % and compare as float
        cleaned = str(value).replace('%', '').strip()
        try:
            return float(cleaned)
        except:
            return value
    
    return str(value).strip()


def compare_results(extracted: dict, expected: dict, image_name: str):
    """Compare extracted results with expected output."""
    print(f"\n{'='*60}")
    print(f"Comparison for {Path(image_name).name}")
    print(f"{'='*60}")
    
    # Field mappings from schema to expected format
    field_mappings = {
        'buyer_name': ('buyer', 'name'),
        'street_address': ('buyer', 'address'),
        'phone_number': ('buyer', 'phone'),
        'co_buyer_name': ('co_buyer', 'name'),
        'quantity': ('purchase', 'quantity'),
        'items_purchased': ('purchase', 'items'),
        'make_or_model': ('purchase', 'make_model'),
        'amount_financed': ('financial', 'amount_financed'),
        'finance_charge': ('financial', 'finance_charge'),
        'apr': ('financial', 'apr'),
        'total_of_payments': ('financial', 'total_of_payments'),
        'number_of_payments': ('payment', 'number_of_payments'),
        'amount_of_payments': ('payment', 'payment_amount'),
    }
    
    matches = 0
    total = 0
    
    for schema_field, (expected_group, expected_field) in field_mappings.items():
        if expected_group not in expected:
            continue
        
        expected_value = expected[expected_group].get(expected_field)
        extracted_value = extracted.get(schema_field)
        
        total += 1
        
        # Normalize for comparison
        if expected_field in ['phone', 'address', 'name', 'items', 'make_model', 'payment_amount']:
            norm_expected = normalize_value(expected_value, "string")
            norm_extracted = normalize_value(extracted_value, "string")
        elif expected_field in ['amount_financed', 'finance_charge', 'total_of_payments', 'payment_amount']:
            norm_expected = normalize_value(expected_value, "currency")
            norm_extracted = normalize_value(extracted_value, "currency")
        elif expected_field == 'apr':
            norm_expected = normalize_value(expected_value, "percentage")
            norm_extracted = normalize_value(extracted_value, "percentage")
        else:
            norm_expected = expected_value
            norm_extracted = extracted_value
        
        # Compare
        if norm_expected == norm_extracted:
            matches += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {schema_field:25s}")
        print(f"    Expected: {expected_value}")
        print(f"    Extracted: {extracted_value}")
        if norm_expected != norm_extracted:
            print(f"    (Normalized: {norm_expected} vs {norm_extracted})")
    
    print(f"\nMatch rate: {matches}/{total} ({matches/total*100:.1f}%)")
    return matches, total


def test_pipeline():
    """Test the full extraction pipeline."""
    print("=" * 60)
    print("Full Extraction Pipeline Test")
    print("=" * 60)
    
    # Initialize pipeline
    openai_key = os.getenv("OPENAI_API_KEY")
    pipeline = ExtractionPipeline(
        credentials_path=CREDENTIALS_PATH,
        openai_api_key=openai_key,
        force_openai=False  # Use OpenAI only if confidence is low
    )
    
    if openai_key:
        print("\n✓ OpenAI processor available")
    else:
        print("\n⚠ OpenAI API key not set - will use deterministic extraction only")
    
    results_summary = {}
    
    # Test each image
    for image_path in TEST_IMAGES:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        if not Path(image_path).exists():
            print(f"✗ Image not found: {image_path}")
            continue
        
        try:
            # Run pipeline
            result = pipeline.extract(image_path, save_raw_ocr=True)
            
            # Display results
            extracted_dict = result.schema.to_json_dict()
            
            print(f"\n✓ Extraction complete!")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Used OpenAI: {result.used_openai}")
            print(f"  OCR confidence: {result.confidence_scores}")
            
            print(f"\nExtracted Fields:")
            for field_name, value in extracted_dict.items():
                status = "✓" if value is not None else "✗"
                print(f"  {status} {field_name:25s}: {value}")
            
            # Compare with expected
            expected = EXPECTED_OUTPUTS.get(image_path, {})
            if expected:
                matches, total = compare_results(extracted_dict, expected, image_path)
                results_summary[image_path] = {
                    'matches': matches,
                    'total': total,
                    'match_rate': matches / total if total > 0 else 0
                }
            
            # Save results
            output_path = f"output/pipeline_result_{Path(image_path).stem}.json"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            print(f"\n✓ Results saved to: {output_path}")
            
        except Exception as e:
            print(f"\n✗ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Test Summary")
    print(f"{'='*60}")
    for image_path, summary in results_summary.items():
        print(f"{Path(image_path).name}: {summary['matches']}/{summary['total']} fields matched ({summary['match_rate']*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("Pipeline test complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_pipeline()

