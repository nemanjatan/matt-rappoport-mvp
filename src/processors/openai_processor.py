"""OpenAI integration for improving extraction quality when OCR confidence is low."""

import os
import json
from typing import Dict, List, Optional, Any
from decimal import Decimal

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from src.ocr import OCRResult
from src.schema import InstallmentAgreementSchema


class OpenAIProcessor:
    """OpenAI processor for improving extraction quality."""
    
    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.85  # Use OpenAI if mean confidence below this
    LOW_WORD_CONFIDENCE_THRESHOLD = 0.80  # Use OpenAI if min word confidence below this
    LOW_CONFIDENCE_WORD_RATIO = 0.20  # Use OpenAI if more than 20% of words below threshold
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", use_vision: bool = True):
        """
        Initialize OpenAI processor.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o-mini, supports vision)
            use_vision: If True, use vision model to analyze image directly (default: True)
        
        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.use_vision = use_vision
    
    def should_use_openai(self, ocr_result: OCRResult) -> bool:
        """
        Determine if OpenAI should be used based on OCR confidence.
        
        Args:
            ocr_result: OCR result from VisionOCRClient
        
        Returns:
            True if OpenAI should be used
        """
        # Check overall confidence
        if ocr_result.confidence_scores:
            word_level = ocr_result.confidence_scores.get('word_level', {})
            block_level = ocr_result.confidence_scores.get('block_level', {})
            
            word_mean = word_level.get('mean', 1.0)
            word_min = word_level.get('min', 1.0)
            block_mean = block_level.get('mean', 1.0)
            
            # Use OpenAI if confidence is low
            if word_mean < self.LOW_CONFIDENCE_THRESHOLD:
                return True
            if word_min < self.LOW_WORD_CONFIDENCE_THRESHOLD:
                return True
            if block_mean < self.LOW_CONFIDENCE_THRESHOLD:
                return True
        
        # Check individual word confidences
        if ocr_result.word_annotations:
            low_confidence_count = sum(
                1 for word in ocr_result.word_annotations
                if word.get('confidence', 1.0) < self.LOW_WORD_CONFIDENCE_THRESHOLD
            )
            low_confidence_ratio = low_confidence_count / len(ocr_result.word_annotations)
            if low_confidence_ratio > self.LOW_CONFIDENCE_WORD_RATIO:
                return True
        
        # Check for warnings
        if ocr_result.warnings:
            return True
        
        return False
    
    def extract_from_image_and_ocr(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_format: Optional[str] = None,
        ocr_result: OCRResult = None
    ) -> InstallmentAgreementSchema:
        """
        Extract all fields directly from image and OCR text using OpenAI Vision.
        
        This is the primary extraction method that uses both visual and textual information.
        
        Args:
            image_path: Path to the image file (if provided)
            image_bytes: Image file bytes (if provided instead of image_path)
            image_format: Image format (PNG, JPEG) - required if image_bytes provided
            ocr_result: OCR result from Google Cloud Vision
        
        Returns:
            InstallmentAgreementSchema with extracted values
        """
        import base64
        from pathlib import Path
        
        # Read and encode image
        if image_path:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine image format
            image_ext = Path(image_path).suffix.lower()
            if image_ext == '.png':
                mime_type = 'image/png'
            elif image_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # Default
        elif image_bytes:
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            if image_format:
                mime_type = f'image/{image_format.lower()}'
            else:
                mime_type = 'image/png'  # Default
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        
        # Build prompt with OCR text and instructions
        prompt = self._build_vision_prompt(ocr_result)
        
        # Log OpenAI request
        try:
            from src.utils import get_logger, log_openai_request
            logger = get_logger()
            log_openai_request(logger, prompt, self.model, redact=True)
        except ImportError:
            logger = None
        
        # Call OpenAI Vision API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,  # Deterministic output
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Log OpenAI response
        if logger:
            try:
                from src.utils import log_openai_response
                log_openai_response(logger, response_text, redact=True)
            except ImportError:
                pass
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned invalid JSON: {e}\nResponse text: {response_text[:500]}")
        
        # Create schema from response
        return self._parse_openai_response(response_data, InstallmentAgreementSchema())
    
    def improve_extraction(
        self,
        ocr_result: OCRResult,
        initial_schema: InstallmentAgreementSchema,
        candidate_values: Optional[Dict[str, List[str]]] = None
    ) -> InstallmentAgreementSchema:
        """
        Use OpenAI to improve extraction quality.
        
        Args:
            ocr_result: OCR result with full text
            initial_schema: Initial extraction from deterministic extractor
            candidate_values: Optional dict of field_name -> list of candidate values
        
        Returns:
            Improved InstallmentAgreementSchema
        """
        # Prepare prompt
        prompt = self._build_prompt(ocr_result, initial_schema, candidate_values)
        
        # Log OpenAI request
        try:
            from src.utils import get_logger, log_openai_request
            logger = get_logger()
            log_openai_request(logger, prompt, self.model, redact=True)
        except ImportError:
            logger = None
        
        # Call OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Deterministic output
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Log OpenAI response
        if logger:
            try:
                from src.utils import log_openai_response
                log_openai_response(logger, response_text, redact=True)
            except ImportError:
                pass
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove closing ```
        response_text = response_text.strip()
        
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned invalid JSON: {e}\nResponse text: {response_text[:500]}")
        
        # Validate and create schema
        return self._parse_openai_response(response_data, initial_schema)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for OpenAI."""
        return """You are a precise data extraction assistant for installment credit agreement documents.

CRITICAL RULES - FOLLOW STRICTLY:
1. NEVER hallucinate, invent, or guess data. Only extract values that are EXPLICITLY visible in the OCR text.
2. If a field value is not clearly visible, ambiguous, or partially occluded, return null for that field.
3. DO NOT infer or calculate values unless the calculation is explicitly shown in the document.
4. Strictly conform to the provided JSON schema. Do not add extra fields or modify field names.
5. Normalize values according to these exact rules:
   - Currency fields (amount_financed, finance_charge, total_of_payments, amount_of_payments): 
     Remove $ and commas, use decimal format as number (e.g., 3644.28, not "3644.28")
   - Phone numbers: Format as string "XXX-XXX-XXXX" (e.g., "843-333-4540")
   - APR: Remove % symbol, use decimal number (e.g., 21 for 21%, 0.0 for 0.00%)
   - Integers (quantity, number_of_payments): Use plain integer (e.g., 6, not 6.0)
   - Strings: Use plain strings, trim whitespace
6. Validate numeric relationships ONLY if both values are clearly visible (e.g., if total_of_payments and amount_of_payments * number_of_payments don't match, use the explicitly stated total_of_payments).
7. For make_or_model: Return null if value is "N/A", "NA", or blank. Otherwise return the string value.
8. When in doubt about ANY field, return null. It is better to return null than to guess.

Return ONLY valid JSON matching the exact schema structure. Use null (not "null" string) for missing values."""
    
    def _build_prompt(
        self,
        ocr_result: OCRResult,
        initial_schema: InstallmentAgreementSchema,
        candidate_values: Optional[Dict[str, List[str]]]
    ) -> str:
        """Build the user prompt for OpenAI."""
        prompt_parts = []
        
        # OCR text - prioritize seller section if it exists
        ocr_text = ocr_result.full_text
        # Check if seller information is in the text
        seller_section_start = ocr_text.lower().find('seller')
        if seller_section_start >= 0:
            # Include seller section and surrounding context (first 10000 chars to ensure seller info is included)
            seller_context_start = max(0, seller_section_start - 500)
            seller_context_end = min(len(ocr_text), seller_section_start + 2000)
            seller_section = ocr_text[seller_context_start:seller_context_end]
            # Also include beginning of document
            beginning = ocr_text[:min(3000, seller_context_start)]
            prompt_parts.append("=== OCR TEXT (Seller section highlighted) ===")
            prompt_parts.append(beginning)
            prompt_parts.append("\n--- SELLER SECTION (pay special attention to this) ---")
            prompt_parts.append(seller_section)
            if len(ocr_text) > seller_context_end:
                prompt_parts.append(f"\n[Text continues. Total length: {len(ocr_text)} characters]")
        else:
            prompt_parts.append("=== OCR TEXT ===")
            prompt_parts.append(ocr_text[:10000])  # Increased limit to ensure seller info is included
            if len(ocr_text) > 10000:
                prompt_parts.append(f"\n[Text truncated. Total length: {len(ocr_text)} characters]")
        
        # Initial extraction
        prompt_parts.append("\n=== INITIAL EXTRACTION (may contain errors) ===")
        initial_dict = initial_schema.to_json_dict()
        prompt_parts.append(json.dumps(initial_dict, indent=2))
        
        # Candidate values if provided
        if candidate_values:
            prompt_parts.append("\n=== CANDIDATE VALUES (from proximity search) ===")
            for field_name, candidates in candidate_values.items():
                if candidates:
                    prompt_parts.append(f"{field_name}: {', '.join(candidates[:5])}")  # Limit to top 5
        
        # Confidence scores
        if ocr_result.confidence_scores:
            prompt_parts.append("\n=== OCR CONFIDENCE SCORES ===")
            prompt_parts.append(json.dumps(ocr_result.confidence_scores, indent=2))
        
        # Instructions
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append("""
Please review the OCR text and improve the extraction:

1. Normalize all values according to schema rules
2. Resolve ambiguities by using context from the full document
3. Validate numeric relationships (e.g., check if total_of_payments = amount_of_payments * number_of_payments)
4. Fill missing fields ONLY if the value is clearly visible in the OCR text - this is especially important for seller fields
5. If unsure about any field, return null - DO NOT GUESS
6. CRITICAL: If seller fields are missing or null, you MUST extract them from the OCR text. Seller information appears in the SELLER section, which comes BEFORE the BUYER section in the document.
   - Seller name: Look for text near "SELLER" or "Seller's Name" (e.g., "PASSANANTES HOME FOOD SERVICES")
   - Seller address: Look for address near "Seller's Address" label (e.g., "1901 FARRAGUT AVENUE")
   - Seller city, state, zip: These appear near the seller address. Look for text like "BRISTOL, PA 19007" near the seller section. DO NOT use buyer city/state/zip (like "LIBERTY, SC 29657").
   - Seller phone: Look for phone number near "Seller's Phone Number" label (e.g., "800-772-7786"). DO NOT use buyer phone numbers.
7. CRITICAL DISTINCTION: The document has TWO sections:
   - SELLER section: Contains seller_name, seller_address, seller_city, seller_state, seller_zip_code, seller_phone_number
   - BUYER section: Contains buyer_name, buyer_address, buyer_phone_number, co_buyer_name, co_buyer_address, co_buyer_phone_number
   DO NOT mix them up! If you see "BRISTOL, PA 19007" near seller address "1901 FARRAGUT AVENUE", that is seller information. If you see "LIBERTY, SC 29657" near buyer address "214 Cheyenne Trail", that is buyer information.
8. IMPORTANT: Extract address and phone number for BOTH buyer and co-buyer separately. They may have different addresses and phone numbers.
9. Pay special attention to the "SELLER SECTION" in the OCR text if it is highlighted.

Return a JSON object with this exact structure (include ALL fields, including seller fields):
{
  "seller_name": "string or null",
  "seller_address": "string or null",
  "seller_city": "string or null",
  "seller_state": "string or null (e.g., PA, SC)",
  "seller_zip_code": "string or null",
  "seller_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "buyer_name": "string or null",
  "buyer_address": "string or null",
  "buyer_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "co_buyer_name": "string or null",
  "co_buyer_address": "string or null",
  "co_buyer_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "quantity": integer or null,
  "items_purchased": "string or null",
  "make_or_model": "string or null (use null for N/A)",
  "amount_financed": number or null (decimal, no $ or commas),
  "finance_charge": number or null (decimal, no $ or commas),
  "apr": number or null (decimal, no % symbol),
  "total_of_payments": number or null (decimal, no $ or commas),
  "number_of_payments": integer or null,
  "amount_of_payments": number or null (decimal, no $ or commas)
}
""")
        
        return "\n".join(prompt_parts)
    
    def _build_vision_prompt(self, ocr_result: OCRResult) -> str:
        """Build prompt for vision-based extraction."""
        prompt_parts = []
        
        prompt_parts.append("=== OCR TEXT FROM GOOGLE CLOUD VISION ===")
        prompt_parts.append(ocr_result.full_text[:10000])  # Include more context for vision
        if len(ocr_result.full_text) > 10000:
            prompt_parts.append(f"\n[Text truncated. Total length: {len(ocr_result.full_text)} characters]")
        
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append("""
You are analyzing an installment credit agreement document. You have access to:
1. The actual image of the document (visible above)
2. The OCR text extracted by Google Cloud Vision (shown above)

Your task is to extract ALL fields from this document. Use BOTH the visual layout of the document AND the OCR text to accurately extract information.

CRITICAL RULES:
1. Use the visual layout to understand document structure - seller information appears in the SELLER section, buyer information in the BUYER section
2. Use OCR text to get exact text values, but verify against the visual document
3. Pay attention to spatial relationships - fields are often near their labels
4. NEVER hallucinate or guess - only extract what you can clearly see
5. Normalize values according to schema rules:
   - Currency: Remove $ and commas, use decimal (e.g., 3644.28)
   - Phone: Format as XXX-XXX-XXXX
   - APR: Remove % symbol, use decimal (e.g., 21 for 21%)
   - Integers: Use plain integer (e.g., 6)
   - Strings: Trim whitespace

SELLER INFORMATION:
- Look for the SELLER section (usually appears before BUYER section)
- Seller name: Near "SELLER" or "Seller's Name" label
- Seller address: Near "Seller's Address" label (e.g., "1901 FARRAGUT AVENUE")
- Seller city, state, zip: Near seller address (e.g., "BRISTOL, PA 19007")
- Seller phone: Near "Seller's Phone Number" label (e.g., "800-772-7786")
- IMPORTANT: Do NOT confuse seller information with buyer information

BUYER INFORMATION:
- Look for the BUYER section (usually appears after SELLER section)
- Buyer 1: buyer_name, buyer_address, buyer_phone_number (near "Buyer 1's Name", "Buyer 1's Address", "Buyer 1's Phone Number")
- Buyer 2 (Co-Buyer): co_buyer_name, co_buyer_address, co_buyer_phone_number (near "Buyer 2's Name", "Buyer 2's Address", "Buyer 2's Phone Number")
- IMPORTANT: Extract address and phone number for BOTH buyer and co-buyer separately

Return a JSON object with this exact structure (include ALL fields):
{
  "seller_name": "string or null",
  "seller_address": "string or null",
  "seller_city": "string or null",
  "seller_state": "string or null (e.g., PA, SC)",
  "seller_zip_code": "string or null",
  "seller_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "buyer_name": "string or null",
  "buyer_address": "string or null",
  "buyer_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "co_buyer_name": "string or null",
  "co_buyer_address": "string or null",
  "co_buyer_phone_number": "string or null (format: XXX-XXX-XXXX)",
  "quantity": integer or null,
  "items_purchased": "string or null",
  "make_or_model": "string or null (use null for N/A)",
  "amount_financed": number or null (decimal, no $ or commas),
  "finance_charge": number or null (decimal, no $ or commas),
  "apr": number or null (decimal, no % symbol),
  "total_of_payments": number or null (decimal, no $ or commas),
  "number_of_payments": integer or null,
  "amount_of_payments": number or null (decimal, no $ or commas)
}
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_openai_response(
        self,
        response_data: Dict[str, Any],
        initial_schema: InstallmentAgreementSchema
    ) -> InstallmentAgreementSchema:
        """
        Parse OpenAI response and create schema.
        
        Args:
            response_data: Parsed JSON from OpenAI
            initial_schema: Initial schema (used as fallback for validation)
        
        Returns:
            InstallmentAgreementSchema with improved values
        """
        # Clean and validate the response
        cleaned_data = {}
        
        for field_name in InstallmentAgreementSchema.model_fields.keys():
            value = response_data.get(field_name)
            
            # If OpenAI returned null, None, empty string, or the field is missing, use None
            if value is None or value == "null" or value == "" or (isinstance(value, str) and value.strip() == ""):
                cleaned_data[field_name] = None
            else:
                # Handle string "null" explicitly
                if isinstance(value, str) and value.lower() == "null":
                    cleaned_data[field_name] = None
                else:
                    cleaned_data[field_name] = value
        
        # Create schema (will validate and normalize)
        try:
            return InstallmentAgreementSchema(**cleaned_data)
        except Exception as e:
            # If validation fails, log and return initial schema
            print(f"Warning: OpenAI response validation failed: {e}")
            print(f"OpenAI response: {json.dumps(response_data, indent=2)}")
            return initial_schema

