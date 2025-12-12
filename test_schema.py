"""Test script to validate the canonical schema."""

import json
from decimal import Decimal
from src.schema import InstallmentAgreementSchema, FieldTypes


def test_schema_validation():
    """Test schema validation with various input formats."""
    
    print("=" * 60)
    print("Testing InstallmentAgreementSchema")
    print("=" * 60)
    
    # Test 1: Complete valid data
    print("\n1. Testing complete valid data:")
    valid_data = {
        "buyer_name": "John Doe",
        "co_buyer_name": "Jane Doe",
        "street_address": "123 Main St, City, ST 12345",
        "phone_number": "843-333-4540",
        "quantity": 2,
        "items_purchased": "Appliances",
        "make_or_model": "Platinum Couture / Cutler",
        "amount_financed": "3644.28",
        "finance_charge": "0.00",
        "apr": "21",
        "total_of_payments": "3644.28",
        "number_of_payments": 6,
        "amount_of_payments": "607.38"
    }
    
    try:
        schema = InstallmentAgreementSchema(**valid_data)
        print("✓ Valid data accepted")
        print(f"  APR: {schema.apr} (type: {type(schema.apr).__name__})")
        print(f"  Amount Financed: {schema.amount_financed} (type: {type(schema.amount_financed).__name__})")
        print(f"  Phone: {schema.phone_number}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Currency normalization
    print("\n2. Testing currency normalization:")
    currency_tests = [
        ("$3,644.28", Decimal("3644.28")),
        ("3,644.28", Decimal("3644.28")),
        ("3644.28", Decimal("3644.28")),
        ("$0.00", Decimal("0.00")),
    ]
    
    for input_val, expected in currency_tests:
        try:
            schema = InstallmentAgreementSchema(amount_financed=input_val)
            if schema.amount_financed == expected:
                print(f"✓ '{input_val}' -> {schema.amount_financed}")
            else:
                print(f"✗ '{input_val}' -> {schema.amount_financed} (expected {expected})")
        except Exception as e:
            print(f"✗ '{input_val}' -> Error: {e}")
    
    # Test 3: Phone number normalization
    print("\n3. Testing phone number normalization:")
    phone_tests = [
        ("843-333-4540", "843-333-4540"),
        ("(843) 333-4540", "843-333-4540"),
        ("8433334540", "843-333-4540"),
        ("18433334540", "843-333-4540"),  # 11 digits starting with 1
    ]
    
    for input_val, expected in phone_tests:
        try:
            schema = InstallmentAgreementSchema(phone_number=input_val)
            if schema.phone_number == expected:
                print(f"✓ '{input_val}' -> {schema.phone_number}")
            else:
                print(f"✗ '{input_val}' -> {schema.phone_number} (expected {expected})")
        except Exception as e:
            print(f"✗ '{input_val}' -> Error: {e}")
    
    # Test 4: APR normalization
    print("\n4. Testing APR normalization:")
    apr_tests = [
        ("21%", Decimal("21")),
        ("21", Decimal("21")),
        ("0.00%", Decimal("0.00")),
        ("0.000%", Decimal("0.000")),
    ]
    
    for input_val, expected in apr_tests:
        try:
            schema = InstallmentAgreementSchema(apr=input_val)
            if schema.apr == expected:
                print(f"✓ '{input_val}' -> {schema.apr}")
            else:
                print(f"✗ '{input_val}' -> {schema.apr} (expected {expected})")
        except Exception as e:
            print(f"✗ '{input_val}' -> Error: {e}")
    
    # Test 5: N/A handling
    print("\n5. Testing N/A value handling:")
    na_tests = [
        ("N/A", None),
        ("NA", None),
        ("", None),
        ("n/a", None),
    ]
    
    for input_val, expected in na_tests:
        try:
            schema = InstallmentAgreementSchema(make_or_model=input_val)
            if schema.make_or_model == expected:
                print(f"✓ '{input_val}' -> {schema.make_or_model}")
            else:
                print(f"✗ '{input_val}' -> {schema.make_or_model} (expected {expected})")
        except Exception as e:
            print(f"✗ '{input_val}' -> Error: {e}")
    
    # Test 6: Null/None values
    print("\n6. Testing null/None values:")
    try:
        schema = InstallmentAgreementSchema()
        print("✓ Empty schema created (all fields None)")
        print(f"  Fields: {len([f for f in schema.model_fields.keys()])} total")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 7: Partial data
    print("\n7. Testing partial data:")
    partial_data = {
        "buyer_name": "David Powers",
        "phone_number": "843-333-4540",
        "quantity": 1,
        "items_purchased": "Food / Goods and/or Services",
        "make_or_model": "N/A",
        "amount_financed": "$3,644.28",
        "finance_charge": "$0.00",
        "apr": "0.00%",
        "total_of_payments": "$3,644.28",
        "number_of_payments": 6,
        "amount_of_payments": "$607.38"
    }
    
    try:
        schema = InstallmentAgreementSchema(**partial_data)
        print("✓ Partial data accepted")
        print(f"  Make/Model: {schema.make_or_model} (should be None)")
        print(f"  Co-buyer: {schema.co_buyer_name} (should be None)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 8: JSON serialization
    print("\n8. Testing JSON serialization:")
    try:
        schema = InstallmentAgreementSchema(**valid_data)
        json_dict = schema.to_json_dict()
        json_str = json.dumps(json_dict, indent=2)
        print("✓ JSON serialization successful")
        print(f"  Sample: {json_str[:200]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 9: Field types
    print("\n9. Testing field type definitions:")
    try:
        field_types = FieldTypes()
        all_fields = FieldTypes.get_all_fields()
        print(f"✓ Field types defined for {len(all_fields)} fields")
        for field in all_fields[:5]:  # Show first 5
            field_type = FieldTypes.get_field_type(field)
            print(f"  {field}: {field_type.__name__}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 10: Integer normalization
    print("\n10. Testing integer normalization:")
    int_tests = [
        ("2", 2),
        (2, 2),
        ("3.0", 3),
        (3.0, 3),
    ]
    
    for input_val, expected in int_tests:
        try:
            schema = InstallmentAgreementSchema(quantity=input_val)
            if schema.quantity == expected:
                print(f"✓ '{input_val}' -> {schema.quantity}")
            else:
                print(f"✗ '{input_val}' -> {schema.quantity} (expected {expected})")
        except Exception as e:
            print(f"✗ '{input_val}' -> Error: {e}")
    
    print("\n" + "=" * 60)
    print("Schema validation tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_schema_validation()

