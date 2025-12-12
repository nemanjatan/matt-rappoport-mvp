"""Tests for error handling and pipeline robustness."""

import pytest
from src.ocr import OCRResult
from src.extractors import DeterministicExtractor
from src.schema import InstallmentAgreementSchema


class TestMalformedOCR:
    """Test that pipeline handles malformed OCR gracefully."""
    
    def test_empty_ocr_text(self):
        """Test handling of empty OCR text."""
        empty_result = OCRResult(
            full_text="",
            word_annotations=[],
            block_annotations=[],
            confidence_scores={},
            raw_response={},
            warnings=[]
        )
        
        extractor = DeterministicExtractor(empty_result)
        schema = extractor.extract_all_fields()
        
        # Should return schema with all None values, not crash
        assert schema is not None, "Should return schema even with empty OCR"
        assert isinstance(schema, InstallmentAgreementSchema), "Should return valid schema"
        
        # All fields should be None
        for field_name in InstallmentAgreementSchema.model_fields.keys():
            value = getattr(schema, field_name)
            assert value is None, f"{field_name} should be None for empty OCR"
    
    def test_malformed_currency_values(self):
        """Test handling of malformed currency values in OCR."""
        # Create OCR result with malformed currency
        malformed_result = OCRResult(
            full_text="Amount Financed: $abc,def.ghi",
            word_annotations=[
                {"text": "Amount", "bounding_box": [{"x": 0, "y": 0}], "confidence": 0.9},
                {"text": "Financed", "bounding_box": [{"x": 50, "y": 0}], "confidence": 0.9},
                {"text": "$abc,def.ghi", "bounding_box": [{"x": 150, "y": 0}], "confidence": 0.8}
            ],
            block_annotations=[],
            confidence_scores={},
            raw_response={},
            warnings=[]
        )
        
        extractor = DeterministicExtractor(malformed_result)
        schema = extractor.extract_all_fields()
        
        # Should not crash, should return None for malformed values
        assert schema is not None, "Should handle malformed currency gracefully"
        # Amount financed should be None (can't parse malformed value)
        # This is acceptable - better to return None than crash
    
    def test_missing_labels(self):
        """Test handling of OCR with missing field labels."""
        # OCR with no recognizable labels
        no_labels_result = OCRResult(
            full_text="Some random text without any field labels or structure",
            word_annotations=[
                {"text": word, "bounding_box": [{"x": i*50, "y": 0}], "confidence": 0.9}
                for i, word in enumerate("Some random text without any field labels".split())
            ],
            block_annotations=[],
            confidence_scores={},
            raw_response={},
            warnings=[]
        )
        
        extractor = DeterministicExtractor(no_labels_result)
        schema = extractor.extract_all_fields()
        
        # Should return schema with None values, not crash
        assert schema is not None, "Should handle missing labels gracefully"
        assert isinstance(schema, InstallmentAgreementSchema), "Should return valid schema"
    
    def test_low_confidence_ocr(self):
        """Test handling of low confidence OCR."""
        low_confidence_result = OCRResult(
            full_text="Buyer's Name: John Doe\nAmount Financed: $1000.00",
            word_annotations=[
                {"text": "Buyer's", "bounding_box": [{"x": 0, "y": 0}], "confidence": 0.3},
                {"text": "Name", "bounding_box": [{"x": 50, "y": 0}], "confidence": 0.3},
                {"text": "John", "bounding_box": [{"x": 100, "y": 0}], "confidence": 0.3},
                {"text": "Doe", "bounding_box": [{"x": 150, "y": 0}], "confidence": 0.3},
            ],
            block_annotations=[],
            confidence_scores={
                "word_level": {"mean": 0.3, "min": 0.3, "max": 0.3},
                "block_level": {"mean": 0.3, "min": 0.3, "max": 0.3}
            },
            raw_response={},
            warnings=["Low confidence detected"]
        )
        
        extractor = DeterministicExtractor(low_confidence_result)
        schema = extractor.extract_all_fields()
        
        # Should still return valid schema
        assert schema is not None, "Should handle low confidence OCR"
        assert isinstance(schema, InstallmentAgreementSchema), "Should return valid schema"


class TestPipelineRobustness:
    """Test pipeline robustness to various edge cases."""
    
    def test_missing_image_file(self, pipeline):
        """Test handling of missing image file."""
        with pytest.raises(FileNotFoundError):
            pipeline.extract("nonexistent_image.png")
    
    def test_invalid_image_format(self, pipeline):
        """Test handling of invalid image format."""
        # Create a dummy file with wrong extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"not an image")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                pipeline.extract(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)
    
    def test_pipeline_returns_valid_result(self, pipeline, img_1805_path):
        """Test that pipeline always returns valid ExtractionResult."""
        result = pipeline.extract(img_1805_path)
        
        # Should have all required attributes
        assert hasattr(result, 'schema'), "Result should have schema"
        assert hasattr(result, 'ocr_result'), "Result should have ocr_result"
        assert hasattr(result, 'used_openai'), "Result should have used_openai"
        assert hasattr(result, 'confidence_scores'), "Result should have confidence_scores"
        assert hasattr(result, 'processing_time'), "Result should have processing_time"
        
        # Schema should be valid
        assert isinstance(result.schema, InstallmentAgreementSchema), \
            "Schema should be InstallmentAgreementSchema instance"
        
        # Processing time should be positive
        assert result.processing_time > 0, "Processing time should be positive"
    
    def test_pipeline_to_dict(self, pipeline, img_1805_path):
        """Test that pipeline result can be converted to dict."""
        result = pipeline.extract(img_1805_path)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict), "to_dict() should return dict"
        assert 'extracted_data' in result_dict, "Dict should have extracted_data"
        assert 'metadata' in result_dict, "Dict should have metadata"
        
        # Metadata should contain expected keys
        metadata = result_dict['metadata']
        assert 'used_openai' in metadata, "Metadata should have used_openai"
        assert 'processing_time_seconds' in metadata, "Metadata should have processing_time_seconds"

