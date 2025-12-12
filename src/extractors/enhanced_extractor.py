"""Enhanced extractor that combines deterministic extraction with OpenAI when needed."""

from typing import Optional, Dict, List
from src.ocr import OCRResult
from src.extractors import DeterministicExtractor
from src.processors import OpenAIProcessor
from src.schema import InstallmentAgreementSchema


class EnhancedExtractor:
    """Enhanced extractor with OpenAI fallback for low-confidence extractions."""
    
    def __init__(
        self,
        ocr_result: OCRResult,
        openai_processor: Optional[OpenAIProcessor] = None,
        force_openai: bool = False
    ):
        """
        Initialize enhanced extractor.
        
        Args:
            ocr_result: OCR result from VisionOCRClient
            openai_processor: Optional OpenAI processor. If None, OpenAI won't be used.
            force_openai: If True, always use OpenAI regardless of confidence
        """
        self.ocr_result = ocr_result
        self.openai_processor = openai_processor
        self.force_openai = force_openai
        
        # Create deterministic extractor
        self.deterministic_extractor = DeterministicExtractor(ocr_result)
    
    def extract_all_fields(self) -> InstallmentAgreementSchema:
        """
        Extract all fields using deterministic extraction, with OpenAI fallback if needed.
        
        Returns:
            InstallmentAgreementSchema with extracted values
        """
        try:
            from src.utils import get_logger, log_openai_usage, log_field_extraction
            logger = get_logger()
        except ImportError:
            logger = None
        
        # Step 1: Try deterministic extraction
        initial_schema = self.deterministic_extractor.extract_all_fields()
        
        # Step 2: Check if OpenAI should be used
        should_use_openai = False
        reason = None
        
        if self.force_openai:
            should_use_openai = True
            reason = "Force OpenAI enabled"
        elif self.openai_processor:
            # Check if critical seller fields are missing - use OpenAI to extract them
            seller_fields_missing = (
                not initial_schema.seller_name and
                not initial_schema.seller_address and
                not initial_schema.seller_phone_number
            )
            
            if seller_fields_missing:
                # Check if seller information appears to be in OCR text
                ocr_lower = self.ocr_result.full_text.lower()
                has_seller_info = (
                    'seller' in ocr_lower or
                    'passanante' in ocr_lower or
                    '1901' in ocr_lower and 'farragut' in ocr_lower
                )
                
                if has_seller_info:
                    should_use_openai = True
                    reason = "Seller fields missing but appear in OCR text - using OpenAI to extract"
            
            # Also check confidence-based criteria
            if not should_use_openai:
                should_use_openai = self.openai_processor.should_use_openai(self.ocr_result)
                if should_use_openai:
                    # Determine reason
                    if self.ocr_result.confidence_scores:
                        word_level = self.ocr_result.confidence_scores.get('word_level', {})
                        word_mean = word_level.get('mean', 1.0)
                        if word_mean < self.openai_processor.LOW_CONFIDENCE_THRESHOLD:
                            reason = f"Low OCR confidence: {word_mean:.3f} < {self.openai_processor.LOW_CONFIDENCE_THRESHOLD}"
                    if self.ocr_result.warnings:
                        reason = f"OCR warnings present: {self.ocr_result.warnings}"
                    if not reason:
                        reason = "Low confidence detected"
                else:
                    reason = "OCR confidence sufficient"
        
        # Log OpenAI usage decision
        if logger:
            log_openai_usage(logger, should_use_openai, reason)
        
        # Step 3: Use OpenAI if needed
        if should_use_openai and self.openai_processor:
            try:
                # Collect candidate values for context
                candidate_values = self._collect_candidate_values()
                
                # Improve extraction with OpenAI
                improved_schema = self.openai_processor.improve_extraction(
                    ocr_result=self.ocr_result,
                    initial_schema=initial_schema,
                    candidate_values=candidate_values
                )
                
                # Log improved fields
                if logger:
                    logger.info("=" * 60)
                    logger.info("OPENAI ENHANCED EXTRACTION")
                    logger.info("=" * 60)
                    for field_name in improved_schema.model_fields.keys():
                        initial_value = getattr(initial_schema, field_name)
                        improved_value = getattr(improved_schema, field_name)
                        if initial_value != improved_value:
                            logger.info(f"  ↻ {field_name:25s}: {initial_value} → {improved_value}")
                        else:
                            log_field_extraction(logger, field_name, improved_value, source="openai")
                    logger.info("=" * 60)
                
                return improved_schema
            except Exception as e:
                # If OpenAI fails, fall back to deterministic extraction
                if logger:
                    logger.warning(f"OpenAI processing failed: {e}, falling back to deterministic extraction")
                else:
                    print(f"Warning: OpenAI processing failed: {e}")
                return initial_schema
        
        return initial_schema
    
    def _collect_candidate_values(self) -> Dict[str, List[str]]:
        """Collect candidate values for all fields to provide context to OpenAI."""
        candidate_values = {}
        
        for field_name in InstallmentAgreementSchema.model_fields.keys():
            candidates = self.deterministic_extractor._find_field_candidates(field_name)
            # Get top 5 candidate values
            top_candidates = sorted(
                candidates,
                key=lambda c: (c.distance, -c.confidence)
            )[:5]
            candidate_values[field_name] = [c.value for c in top_candidates]
        
        return candidate_values

