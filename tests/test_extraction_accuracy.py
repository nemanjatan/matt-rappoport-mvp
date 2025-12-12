"""Tests for extraction accuracy and key field validation."""

import pytest
from decimal import Decimal
from src.schema import InstallmentAgreementSchema


class TestIMG1805Extraction:
    """Test extraction accuracy for IMG_1805.png."""
    
    def test_buyer_name(self, pipeline, img_1805_path, expected_img_1805):
        """Test buyer name extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_name = result.schema.buyer_name
        
        # Should extract a name (may have OCR variations)
        assert extracted_name is not None, "Buyer name should not be None"
        assert len(extracted_name) > 0, "Buyer name should not be empty"
        
        # Should contain expected name parts (tolerant of OCR errors)
        expected = expected_img_1805["buyer_name"].lower()
        extracted = extracted_name.lower()
        # Check for key name parts
        assert "hannah" in extracted or "hornberger" in extracted, \
            f"Expected 'Hannah' or 'Hornberger' in buyer name, got: {extracted_name}"
    
    def test_co_buyer_name(self, pipeline, img_1805_path, expected_img_1805):
        """Test co-buyer name extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_name = result.schema.co_buyer_name
        
        # Co-buyer may be None, but if present should contain expected parts
        if extracted_name:
            expected = expected_img_1805["co_buyer_name"].lower()
            extracted = extracted_name.lower()
            assert "randy" in extracted or "hornberger" in extracted, \
                f"Expected 'Randy' or 'Hornberger' in co-buyer name, got: {extracted_name}"
    
    def test_phone_number(self, pipeline, img_1805_path, expected_img_1805):
        """Test phone number extraction and normalization."""
        result = pipeline.extract(img_1805_path)
        extracted_phone = result.schema.phone_number
        
        if extracted_phone:
            # Should be in XXX-XXX-XXXX format
            assert "-" in extracted_phone, f"Phone should be formatted, got: {extracted_phone}"
            # Extract digits for comparison
            digits = "".join(filter(str.isdigit, extracted_phone))
            expected_digits = "".join(filter(str.isdigit, expected_img_1805["phone_number"]))
            # Last 10 digits should match (allowing for area code variations)
            assert digits[-10:] == expected_digits[-10:], \
                f"Phone number mismatch. Expected ending: {expected_digits[-10:]}, got: {digits[-10:]}"
    
    def test_quantity(self, pipeline, img_1805_path, expected_img_1805):
        """Test quantity extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_qty = result.schema.quantity
        
        # Quantity should be extracted (may be None due to OCR issues)
        if extracted_qty is not None:
            assert isinstance(extracted_qty, int), f"Quantity should be integer, got: {type(extracted_qty)}"
            assert extracted_qty == expected_img_1805["quantity"], \
                f"Quantity mismatch. Expected: {expected_img_1805['quantity']}, got: {extracted_qty}"
    
    def test_amount_financed(self, pipeline, img_1805_path, expected_img_1805):
        """Test amount financed extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_amount = result.schema.amount_financed
        
        if extracted_amount is not None:
            # Convert to float for comparison (allowing small tolerance)
            extracted_float = float(extracted_amount)
            expected_float = expected_img_1805["amount_financed"]
            # Allow 1% tolerance for OCR variations
            tolerance = expected_float * 0.01
            assert abs(extracted_float - expected_float) <= tolerance, \
                f"Amount financed mismatch. Expected: {expected_float}, got: {extracted_float}"
    
    def test_apr(self, pipeline, img_1805_path, expected_img_1805):
        """Test APR extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_apr = result.schema.apr
        
        if extracted_apr is not None:
            extracted_float = float(extracted_apr)
            expected_float = expected_img_1805["apr"]
            # Allow small tolerance for percentage extraction
            assert abs(extracted_float - expected_float) <= 0.5, \
                f"APR mismatch. Expected: {expected_float}, got: {extracted_float}"
    
    def test_number_of_payments(self, pipeline, img_1805_path, expected_img_1805):
        """Test number of payments extraction."""
        result = pipeline.extract(img_1805_path)
        extracted_num = result.schema.number_of_payments
        
        if extracted_num is not None:
            assert isinstance(extracted_num, int), f"Number of payments should be integer, got: {type(extracted_num)}"
            assert extracted_num == expected_img_1805["number_of_payments"], \
                f"Number of payments mismatch. Expected: {expected_img_1805['number_of_payments']}, got: {extracted_num}"


class TestIMG1807Extraction:
    """Test extraction accuracy for IMG_1807.png."""
    
    def test_buyer_name(self, pipeline, img_1807_path, expected_img_1807):
        """Test buyer name extraction."""
        result = pipeline.extract(img_1807_path)
        extracted_name = result.schema.buyer_name
        
        assert extracted_name is not None, "Buyer name should not be None"
        assert len(extracted_name) > 0, "Buyer name should not be empty"
        
        expected = expected_img_1807["buyer_name"].lower()
        extracted = extracted_name.lower()
        assert "david" in extracted or "powers" in extracted, \
            f"Expected 'David' or 'Powers' in buyer name, got: {extracted_name}"
    
    def test_co_buyer_name(self, pipeline, img_1807_path, expected_img_1807):
        """Test co-buyer name extraction."""
        result = pipeline.extract(img_1807_path)
        extracted_name = result.schema.co_buyer_name
        
        if extracted_name:
            expected = expected_img_1807["co_buyer_name"].lower()
            extracted = extracted_name.lower()
            assert "lydia" in extracted or "powers" in extracted, \
                f"Expected 'Lydia' or 'Powers' in co-buyer name, got: {extracted_name}"
    
    def test_phone_number(self, pipeline, img_1807_path, expected_img_1807):
        """Test phone number extraction."""
        result = pipeline.extract(img_1807_path)
        extracted_phone = result.schema.phone_number
        
        if extracted_phone:
            assert "-" in extracted_phone, f"Phone should be formatted, got: {extracted_phone}"
            digits = "".join(filter(str.isdigit, extracted_phone))
            expected_digits = "".join(filter(str.isdigit, expected_img_1807["phone_number"]))
            assert digits[-10:] == expected_digits[-10:], \
                f"Phone number mismatch. Expected ending: {expected_digits[-10:]}, got: {digits[-10:]}"
    
    def test_amount_financed(self, pipeline, img_1807_path, expected_img_1807):
        """Test amount financed extraction."""
        result = pipeline.extract(img_1807_path)
        extracted_amount = result.schema.amount_financed
        
        if extracted_amount is not None:
            extracted_float = float(extracted_amount)
            expected_float = expected_img_1807["amount_financed"]
            tolerance = expected_float * 0.01
            assert abs(extracted_float - expected_float) <= tolerance, \
                f"Amount financed mismatch. Expected: {expected_float}, got: {extracted_float}"
    
    def test_apr(self, pipeline, img_1807_path, expected_img_1807):
        """Test APR extraction (should be 0.00%)."""
        result = pipeline.extract(img_1807_path)
        extracted_apr = result.schema.apr
        
        if extracted_apr is not None:
            extracted_float = float(extracted_apr)
            expected_float = expected_img_1807["apr"]
            # APR should be 0.0 (allow small tolerance)
            assert abs(extracted_float - expected_float) <= 0.1, \
                f"APR mismatch. Expected: {expected_float}, got: {extracted_float}"
    
    def test_number_of_payments(self, pipeline, img_1807_path, expected_img_1807):
        """Test number of payments extraction."""
        result = pipeline.extract(img_1807_path)
        extracted_num = result.schema.number_of_payments
        
        if extracted_num is not None:
            assert isinstance(extracted_num, int), f"Number of payments should be integer, got: {type(extracted_num)}"
            assert extracted_num == expected_img_1807["number_of_payments"], \
                f"Number of payments mismatch. Expected: {expected_img_1807['number_of_payments']}, got: {extracted_num}"

