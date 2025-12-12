"""Full extraction pipeline orchestrating OCR, extraction, and post-processing."""

import os
from typing import Optional, Dict, Any
from pathlib import Path

from src.ocr import VisionOCRClient, OCRResult
from src.extractors import EnhancedExtractor, DeterministicExtractor
from src.processors import OpenAIProcessor
from src.schema import InstallmentAgreementSchema
from src.validators import AIValidator


class ExtractionResult:
    """Result of the extraction pipeline."""
    
    def __init__(
        self,
        schema: InstallmentAgreementSchema,
        ocr_result: OCRResult,
        used_openai: bool,
        confidence_scores: Dict[str, Any],
        processing_time: float,
        validation_result: Optional[Any] = None
    ):
        """
        Initialize extraction result.
        
        Args:
            schema: Final extracted schema
            ocr_result: OCR result from Google Cloud Vision
            used_openai: Whether OpenAI was used
            confidence_scores: OCR confidence scores
            processing_time: Total processing time in seconds
            validation_result: Optional validation result from AI validator
        """
        self.schema = schema
        self.ocr_result = ocr_result
        self.used_openai = used_openai
        self.confidence_scores = confidence_scores
        self.processing_time = processing_time
        self.validation_result = validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'extracted_data': self.schema.to_json_dict(),
            'metadata': {
                'used_openai': self.used_openai,
                'confidence_scores': self.confidence_scores,
                'processing_time_seconds': self.processing_time,
                'ocr_text_length': len(self.ocr_result.full_text),
                'ocr_word_count': len(self.ocr_result.word_annotations)
            }
        }
        
        # Add validation info if available
        if self.validation_result:
            result['validation'] = self.validation_result.to_dict()
        
        return result


class ExtractionPipeline:
    """Full extraction pipeline orchestrating all steps."""
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        force_openai: bool = False
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            force_openai: If True, always use OpenAI regardless of confidence.
        """
        # No initialization needed for time
        
        # Initialize OCR client
        self.ocr_client = VisionOCRClient(credentials_path=credentials_path)
        
        # Initialize OpenAI processor (if available)
        self.openai_processor = None
        self.force_openai = force_openai
        
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                # Use vision by default for better extraction (image + OCR text)
                self.openai_processor = OpenAIProcessor(
                    api_key=openai_api_key,
                    use_vision=True  # Enable vision-based extraction
                )
            except Exception as e:
                print(f"Warning: OpenAI processor initialization failed: {e}")
                print("Continuing without OpenAI enhancement.")
        
        # Initialize AI validator (if OpenAI available)
        self.ai_validator = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.ai_validator = AIValidator(api_key=openai_api_key)
            except Exception as e:
                # Validator is optional, continue without it
                pass
    
    def extract(
        self,
        image_path: str,
        save_raw_ocr: bool = False
    ) -> ExtractionResult:
        """
        Run the full extraction pipeline.
        
        Pipeline steps:
        1. Image Upload (validate)
        2. Google Cloud Vision OCR
        3. Rule-based Extraction
        4. Confidence Evaluation
        5. Optional OpenAI Post-processing
        6. Final Structured Output
        
        Args:
            image_path: Path to image file (PNG or JPG)
            save_raw_ocr: Whether to save raw OCR output
        
        Returns:
            ExtractionResult with extracted data and metadata
        """
        import time
        start_time = time.time()
        
        # Initialize logging
        try:
            from src.utils import setup_logger, get_logger
            logger = setup_logger()
            logger.info("=" * 60)
            logger.info(f"EXTRACTION PIPELINE: {Path(image_path).name}")
            logger.info("=" * 60)
        except ImportError:
            logger = None
        
        # Step 1: Image Upload (validate)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_ext = Path(image_path).suffix.lower()
        if image_ext not in ['.png', '.jpg', '.jpeg']:
            raise ValueError(f"Unsupported image format: {image_ext}. Use PNG or JPG.")
        
        # Step 2: Google Cloud Vision OCR
        if logger:
            logger.info("Step 1: Performing OCR...")
        ocr_result = self.ocr_client.extract_text(
            image_path=image_path,
            save_raw_output=save_raw_ocr
        )
        
        # Step 3: Extraction Strategy
        used_openai = False
        if self.openai_processor and (self.force_openai or self.openai_processor.use_vision):
            # Use OpenAI Vision for direct extraction (image + OCR text)
            if logger:
                logger.info("Step 2: OpenAI Vision extraction (image + OCR text)...")
            try:
                schema = self.openai_processor.extract_from_image_and_ocr(
                    image_path=image_path,
                    ocr_result=ocr_result
                )
                used_openai = True
                if logger:
                    logger.info("✓ OpenAI Vision extraction complete")
            except Exception as e:
                if logger:
                    logger.warning(f"OpenAI Vision extraction failed: {e}, falling back to deterministic extraction")
                # Fall back to deterministic extraction
                if logger:
                    logger.info("Step 2: Rule-based extraction (fallback)...")
                extractor = EnhancedExtractor(
                    ocr_result=ocr_result,
                    openai_processor=self.openai_processor,
                    force_openai=False  # Don't force OpenAI in fallback
                )
                schema = extractor.extract_all_fields()
        else:
            # Use deterministic extraction with optional OpenAI enhancement
            if logger:
                logger.info("Step 2: Rule-based extraction...")
            extractor = EnhancedExtractor(
                ocr_result=ocr_result,
                openai_processor=self.openai_processor,
                force_openai=self.force_openai
            )
            
            # Step 4 & 5: Confidence Evaluation & Optional OpenAI Post-processing
            # (Handled internally by EnhancedExtractor)
            schema = extractor.extract_all_fields()
            
            # Determine if OpenAI was used
            if self.openai_processor:
                if self.force_openai:
                    used_openai = True
                else:
                    used_openai = self.openai_processor.should_use_openai(ocr_result)
        
        # Step 6: AI Validation and Correction (if validator available)
        validation_result = None
        if self.ai_validator:
            if logger:
                logger.info("Step 3: AI validation and correction...")
            try:
                validation_result = self.ai_validator.validate_and_correct(schema, ocr_result)
                if validation_result.used_ai:
                    schema = validation_result.corrected_schema
                    if logger:
                        logger.info(f"  Applied {len(validation_result.corrections_applied)} correction(s)")
                        for correction in validation_result.corrections_applied:
                            logger.info(f"    - {correction}")
                        if validation_result.issues_found:
                            logger.info(f"  Found {len(validation_result.issues_found)} issue(s)")
                else:
                    if logger:
                        logger.info("  No corrections needed")
            except Exception as e:
                if logger:
                    logger.warning(f"AI validation failed: {e}, using original extraction")
        
        # Step 7: Final Structured Output
        processing_time = time.time() - start_time
        
        if logger:
            logger.info("=" * 60)
            logger.info("EXTRACTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Used OpenAI: {used_openai}")
            if validation_result and validation_result.used_ai:
                logger.info(f"AI Validation: Applied {len(validation_result.corrections_applied)} correction(s)")
            logger.info("=" * 60)
        
        return ExtractionResult(
            schema=schema,
            ocr_result=ocr_result,
            used_openai=used_openai,
            confidence_scores=ocr_result.confidence_scores or {},
            processing_time=processing_time,
            validation_result=validation_result
        )
    
    def extract_from_bytes(
        self,
        image_bytes: bytes,
        image_format: str = "PNG"
    ) -> ExtractionResult:
        """
        Extract from image bytes (for use with uploaded files).
        
        Args:
            image_bytes: Image file bytes
            image_format: Image format (PNG, JPEG, etc.)
        
        Returns:
            ExtractionResult with extracted data and metadata
        """
        import tempfile
        import time
        
        start_time = time.time()
        
        # Initialize logging
        try:
            from src.utils import setup_logger, get_logger
            logger = setup_logger()
            logger.info("=" * 60)
            logger.info("EXTRACTION PIPELINE: (from bytes)")
            logger.info("=" * 60)
        except ImportError:
            logger = None
        
        # Step 1: Save to temp file for OCR
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_format.lower()}') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Step 2: Google Cloud Vision OCR
            if logger:
                logger.info("Step 1: Performing OCR...")
            ocr_result = self.ocr_client.extract_text(
                image_path=tmp_path,
                save_raw_output=False
            )
            
            # Step 3: Extraction Strategy
            used_openai = False
            if self.openai_processor and (self.force_openai or self.openai_processor.use_vision):
                # Use OpenAI Vision for direct extraction (image + OCR text)
                if logger:
                    logger.info("Step 2: OpenAI Vision extraction (image + OCR text)...")
                try:
                    schema = self.openai_processor.extract_from_image_and_ocr(
                        image_bytes=image_bytes,
                        image_format=image_format,
                        ocr_result=ocr_result
                    )
                    used_openai = True
                    if logger:
                        logger.info("✓ OpenAI Vision extraction complete")
                except Exception as e:
                    if logger:
                        logger.warning(f"OpenAI Vision extraction failed: {e}, falling back to deterministic extraction")
                    # Fall back to deterministic extraction
                    if logger:
                        logger.info("Step 2: Rule-based extraction (fallback)...")
                    extractor = EnhancedExtractor(
                        ocr_result=ocr_result,
                        openai_processor=self.openai_processor,
                        force_openai=False  # Don't force OpenAI in fallback
                    )
                    schema = extractor.extract_all_fields()
            else:
                # Use deterministic extraction with optional OpenAI enhancement
                if logger:
                    logger.info("Step 2: Rule-based extraction...")
                extractor = EnhancedExtractor(
                    ocr_result=ocr_result,
                    openai_processor=self.openai_processor,
                    force_openai=self.force_openai
                )
                
                # Step 4 & 5: Confidence Evaluation & Optional OpenAI Post-processing
                # (Handled internally by EnhancedExtractor)
                schema = extractor.extract_all_fields()
                
                # Determine if OpenAI was used
                if self.openai_processor:
                    if self.force_openai:
                        used_openai = True
                    else:
                        used_openai = self.openai_processor.should_use_openai(ocr_result)
            
            # Step 6: AI Validation and Correction (if validator available)
            validation_result = None
            if self.ai_validator:
                if logger:
                    logger.info("Step 3: AI validation and correction...")
                try:
                    validation_result = self.ai_validator.validate_and_correct(schema, ocr_result)
                    if validation_result.used_ai:
                        schema = validation_result.corrected_schema
                        if logger:
                            logger.info(f"  Applied {len(validation_result.corrections_applied)} correction(s)")
                            for correction in validation_result.corrections_applied:
                                logger.info(f"    - {correction}")
                            if validation_result.issues_found:
                                logger.info(f"  Found {len(validation_result.issues_found)} issue(s)")
                    else:
                        if logger:
                            logger.info("  No corrections needed")
                except Exception as e:
                    if logger:
                        logger.warning(f"AI validation failed: {e}, using original extraction")
            
            # Step 7: Final Structured Output
            processing_time = time.time() - start_time
            
            if logger:
                logger.info("=" * 60)
                logger.info("EXTRACTION COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Processing time: {processing_time:.2f}s")
                logger.info(f"Used OpenAI: {used_openai}")
                if validation_result and validation_result.used_ai:
                    logger.info(f"AI Validation: Applied {len(validation_result.corrections_applied)} correction(s)")
                logger.info("=" * 60)
            
            return ExtractionResult(
                schema=schema,
                ocr_result=ocr_result,
                used_openai=used_openai,
                confidence_scores=ocr_result.confidence_scores or {},
                processing_time=processing_time,
                validation_result=validation_result
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # Save to temporary file
        suffix = '.png' if image_format.upper() == 'PNG' else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = self.extract(tmp_path, save_raw_ocr=False)
            return result
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

