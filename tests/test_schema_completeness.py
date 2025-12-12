"""Tests for schema completeness and validation."""

import pytest
from src.schema import InstallmentAgreementSchema


class TestSchemaCompleteness:
    """Test that schema is complete and properly structured."""
    
    def test_all_fields_present(self, pipeline, img_1805_path):
        """Test that all schema fields are present in extraction result."""
        result = pipeline.extract(img_1805_path)
        schema = result.schema
        
        # Get all field names from schema
        expected_fields = set(InstallmentAgreementSchema.model_fields.keys())
        actual_fields = set(schema.model_dump().keys())
        
        assert expected_fields == actual_fields, \
            f"Schema fields mismatch. Expected: {expected_fields}, got: {actual_fields}"
    
    def test_schema_types(self, pipeline, img_1805_path):
        """Test that schema field types are correct."""
        result = pipeline.extract(img_1805_path)
        schema = result.schema
        
        # String fields
        string_fields = ['buyer_name', 'co_buyer_name', 'street_address', 
                        'phone_number', 'items_purchased', 'make_or_model']
        for field in string_fields:
            value = getattr(schema, field)
            if value is not None:
                assert isinstance(value, str), \
                    f"{field} should be string or None, got: {type(value)}"
        
        # Integer fields
        integer_fields = ['quantity', 'number_of_payments']
        for field in integer_fields:
            value = getattr(schema, field)
            if value is not None:
                assert isinstance(value, int), \
                    f"{field} should be integer or None, got: {type(value)}"
        
        # Decimal fields
        decimal_fields = ['amount_financed', 'finance_charge', 'apr', 
                         'total_of_payments', 'amount_of_payments']
        for field in decimal_fields:
            value = getattr(schema, field)
            if value is not None:
                from decimal import Decimal
                assert isinstance(value, Decimal), \
                    f"{field} should be Decimal or None, got: {type(value)}"
    
    def test_schema_validation(self, pipeline, img_1805_path):
        """Test that schema validates correctly."""
        result = pipeline.extract(img_1805_path)
        schema = result.schema
        
        # Schema should be valid (no validation errors)
        assert schema is not None, "Schema should not be None"
        
        # Should be able to convert to dict
        schema_dict = schema.to_json_dict()
        assert isinstance(schema_dict, dict), "Schema should convert to dict"
        
        # Should be able to recreate schema from dict
        recreated = InstallmentAgreementSchema(**schema_dict)
        assert recreated is not None, "Should be able to recreate schema from dict"
    
    def test_schema_json_serialization(self, pipeline, img_1805_path):
        """Test that schema can be serialized to JSON."""
        import json
        result = pipeline.extract(img_1805_path)
        schema = result.schema
        
        # Should serialize without errors
        json_str = json.dumps(schema.to_json_dict(), default=str)
        assert len(json_str) > 0, "JSON serialization should produce non-empty string"
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "Parsed JSON should be dict"


class TestSchemaNormalization:
    """Test that schema normalizes values correctly."""
    
    def test_phone_normalization(self, pipeline, img_1805_path):
        """Test phone number normalization."""
        result = pipeline.extract(img_1805_path)
        phone = result.schema.phone_number
        
        if phone is not None:
            # Should be in XXX-XXX-XXXX format
            parts = phone.split('-')
            assert len(parts) == 3, f"Phone should have format XXX-XXX-XXXX, got: {phone}"
            assert all(len(part) == 3 or (len(part) == 4 and parts.index(part) == 2) 
                      for part in parts), \
                f"Phone parts should be 3-3-4 digits, got: {phone}"
    
    def test_currency_normalization(self, pipeline, img_1805_path):
        """Test currency value normalization."""
        result = pipeline.extract(img_1805_path)
        
        # Currency fields should be Decimal, not strings
        currency_fields = ['amount_financed', 'finance_charge', 
                          'total_of_payments', 'amount_of_payments']
        for field in currency_fields:
            value = getattr(result.schema, field)
            if value is not None:
                from decimal import Decimal
                assert isinstance(value, Decimal), \
                    f"{field} should be Decimal, got: {type(value)}"
                # Should be positive or zero
                assert float(value) >= 0, \
                    f"{field} should be non-negative, got: {value}"
    
    def test_apr_normalization(self, pipeline, img_1805_path):
        """Test APR normalization."""
        result = pipeline.extract(img_1805_path)
        apr = result.schema.apr
        
        if apr is not None:
            from decimal import Decimal
            assert isinstance(apr, Decimal), f"APR should be Decimal, got: {type(apr)}"
            # APR should be between 0 and 100
            apr_float = float(apr)
            assert 0 <= apr_float <= 100, \
                f"APR should be between 0 and 100, got: {apr_float}"

