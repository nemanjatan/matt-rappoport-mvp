"""AI-powered validator for normalizing and correcting extracted fields."""

import os
import json
import re
from typing import Dict, List, Optional, Any
from decimal import Decimal

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from src.schema import InstallmentAgreementSchema
from src.ocr import OCRResult


class ValidationIssue:
    """Represents a validation issue found in extracted data."""
    
    def __init__(self, field: str, issue_type: str, description: str, severity: str = "medium"):
        self.field = field
        self.issue_type = issue_type
        self.description = description
        self.severity = severity  # low, medium, high
    
    def __repr__(self):
        return f"ValidationIssue({self.field}, {self.issue_type}, {self.severity})"


class ValidationResult:
    """Result of validation and correction."""
    
    def __init__(
        self,
        corrected_schema: InstallmentAgreementSchema,
        issues_found: List[ValidationIssue],
        corrections_applied: List[str],
        used_ai: bool
    ):
        self.corrected_schema = corrected_schema
        self.issues_found = issues_found
        self.corrections_applied = corrections_applied
        self.used_ai = used_ai
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'corrected_data': self.corrected_schema.to_json_dict(),
            'issues_found': [
                {
                    'field': issue.field,
                    'type': issue.issue_type,
                    'description': issue.description,
                    'severity': issue.severity
                }
                for issue in self.issues_found
            ],
            'corrections_applied': self.corrections_applied,
            'used_ai': self.used_ai
        }


class AIValidator:
    """AI-powered validator for normalizing and correcting extracted data."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize AI validator.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def validate_and_correct(
        self,
        schema: InstallmentAgreementSchema,
        ocr_result: OCRResult
    ) -> ValidationResult:
        """
        Validate and correct extracted schema using AI.
        
        This method:
        1. Detects validation issues (name mismatches, address problems, etc.)
        2. Uses AI to normalize and correct issues
        3. Ensures buyer/co-buyer consistency (shared last names, addresses)
        4. Returns corrected schema with applied fixes
        
        Args:
            schema: Initial extracted schema
            ocr_result: OCR result for context
        
        Returns:
            ValidationResult with corrected schema and issues found
        """
        # Step 1: Detect issues
        issues = self._detect_issues(schema, ocr_result)
        
        # Step 2: Determine if AI correction is needed
        needs_correction = any(
            issue.severity in ['medium', 'high'] 
            for issue in issues
        )
        
        if not needs_correction:
            return ValidationResult(
                corrected_schema=schema,
                issues_found=issues,
                corrections_applied=[],
                used_ai=False
            )
        
        # Step 3: Use AI to correct issues
        try:
            corrected_schema = self._ai_correct(schema, ocr_result, issues)
            corrections = [f"Corrected {issue.field}: {issue.description}" for issue in issues]
            
            return ValidationResult(
                corrected_schema=corrected_schema,
                issues_found=issues,
                corrections_applied=corrections,
                used_ai=True
            )
        except Exception as e:
            # If AI correction fails, return original schema
            try:
                from src.utils import get_logger
                logger = get_logger()
                logger.warning(f"AI validation failed: {e}, returning original schema")
            except ImportError:
                pass
            
            return ValidationResult(
                corrected_schema=schema,
                issues_found=issues,
                corrections_applied=[],
                used_ai=False
            )
    
    def _detect_issues(
        self,
        schema: InstallmentAgreementSchema,
        ocr_result: OCRResult
    ) -> List[ValidationIssue]:
        """Detect validation issues in extracted schema."""
        issues = []
        
        # Check name consistency
        buyer_name = schema.buyer_name
        co_buyer_name = schema.co_buyer_name
        
        if buyer_name and co_buyer_name:
            # Extract last names
            buyer_last = self._extract_last_name(buyer_name)
            co_buyer_last = self._extract_last_name(co_buyer_name)
            
            if buyer_last and co_buyer_last:
                # Check if last names are similar (allowing for OCR errors)
                similarity = self._name_similarity(buyer_last, co_buyer_last)
                if similarity < 0.7:  # Less than 70% similar
                    issues.append(ValidationIssue(
                        field="buyer_name/co_buyer_name",
                        issue_type="last_name_mismatch",
                        description=f"Last names don't match: '{buyer_last}' vs '{co_buyer_last}' (similarity: {similarity:.2f})",
                        severity="high"
                    ))
        
        # Check address quality
        buyer_address = schema.street_address
        if buyer_address:
            # Check for common address issues
            if len(buyer_address.split()) < 2:
                issues.append(ValidationIssue(
                    field="street_address",
                    issue_type="address_too_short",
                    description=f"Address seems incomplete: '{buyer_address}'",
                    severity="high"
                ))
            
            # Check if address looks like it contains phone number parts
            if re.search(r'\d{3}-\d{3}-\d{4}', buyer_address):
                issues.append(ValidationIssue(
                    field="street_address",
                    issue_type="address_contains_phone",
                    description=f"Address appears to contain phone number: '{buyer_address}'",
                    severity="high"
                ))
            
            # Check for missing street number
            if not re.search(r'^\d+', buyer_address):
                issues.append(ValidationIssue(
                    field="street_address",
                    issue_type="missing_street_number",
                    description=f"Address missing street number: '{buyer_address}'",
                    severity="medium"
                ))
        
        # Check co-buyer address consistency
        # (Co-buyer should typically share the same address - handled in AI correction)
        
        # Check for OCR-like errors in names (common character confusions)
        if buyer_name:
            if self._has_ocr_errors(buyer_name):
                issues.append(ValidationIssue(
                    field="buyer_name",
                    issue_type="possible_ocr_error",
                    description=f"Name may contain OCR errors: '{buyer_name}'",
                    severity="medium"
                ))
        
        if co_buyer_name:
            if self._has_ocr_errors(co_buyer_name):
                issues.append(ValidationIssue(
                    field="co_buyer_name",
                    issue_type="possible_ocr_error",
                    description=f"Name may contain OCR errors: '{co_buyer_name}'",
                    severity="medium"
                ))
        
        # Check seller information
        seller_name = schema.seller_name
        seller_address = schema.seller_address
        
        if seller_name:
            if self._has_ocr_errors(seller_name):
                issues.append(ValidationIssue(
                    field="seller_name",
                    issue_type="possible_ocr_error",
                    description=f"Name may contain OCR errors: '{seller_name}'",
                    severity="medium"
                ))
        
        if seller_address:
            # Check seller address quality
            if len(seller_address.split()) < 2:
                issues.append(ValidationIssue(
                    field="seller_address",
                    issue_type="address_too_short",
                    description=f"Seller address seems incomplete: '{seller_address}'",
                    severity="high"
                ))
            
            if not re.search(r'^\d+', seller_address):
                issues.append(ValidationIssue(
                    field="seller_address",
                    issue_type="missing_street_number",
                    description=f"Seller address missing street number: '{seller_address}'",
                    severity="medium"
                ))
        
        return issues
    
    def _extract_last_name(self, full_name: str) -> Optional[str]:
        """Extract last name from full name."""
        if not full_name:
            return None
        
        parts = full_name.strip().split()
        if len(parts) >= 2:
            return parts[-1]  # Last part is typically last name
        return None
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names (simple Levenshtein-like)."""
        if not name1 or not name2:
            return 0.0
        
        name1 = name1.lower()
        name2 = name2.lower()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Check if one contains the other
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # Simple character overlap
        set1 = set(name1)
        set2 = set(name2)
        intersection = set1 & set2
        union = set1 | set2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _has_ocr_errors(self, text: str) -> bool:
        """Check if text likely contains OCR errors."""
        # Common OCR character confusions
        ocr_errors = [
            ('rn', 'm'), ('cl', 'd'), ('ii', 'n'), ('vv', 'w'),
            ('0', 'O'), ('1', 'I'), ('5', 'S'), ('8', 'B')
        ]
        
        text_lower = text.lower()
        for error, correct in ocr_errors:
            if error in text_lower:
                return True
        
        # Check for unusual character patterns
        if re.search(r'[^a-zA-Z\s\-\']', text):
            return True
        
        return False
    
    def _ai_correct(
        self,
        schema: InstallmentAgreementSchema,
        ocr_result: OCRResult,
        issues: List[ValidationIssue]
    ) -> InstallmentAgreementSchema:
        """Use AI to correct detected issues."""
        # Build correction prompt
        prompt = self._build_correction_prompt(schema, ocr_result, issues)
        
        # Log request
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
                    "content": self._get_correction_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Log response
        if logger:
            try:
                from src.utils import log_openai_response
                log_openai_response(logger, response_text, redact=True)
            except ImportError:
                pass
        
        # Remove markdown if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            corrected_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from AI: {e}")
        
        # Create corrected schema
        cleaned_data = {}
        for field_name in InstallmentAgreementSchema.model_fields.keys():
            value = corrected_data.get(field_name)
            if value is None or value == "null" or value == "":
                cleaned_data[field_name] = None
            else:
                cleaned_data[field_name] = value
        
        return InstallmentAgreementSchema(**cleaned_data)
    
    def _get_correction_system_prompt(self) -> str:
        """Get system prompt for AI correction."""
        return """You are a data validation and correction assistant for installment credit agreement documents.

Your task is to:
1. Normalize names, especially ensuring buyer and co-buyer share the same last name when they appear to be a couple
2. Correct address extraction errors (e.g., "Suu" should be "500 Ricky Street")
3. Ensure buyer and co-buyer share the same mailing address when appropriate
4. Fix OCR errors in names (e.g., "Hornberse" → "Hornberger", "Hurnberge" → "Hornberger")
5. Only correct fields where issues are clearly visible in the OCR text
6. Do NOT hallucinate or invent data - only correct what is clearly present

CRITICAL RULES:
- If buyer and co-buyer appear to be a couple (same address, related names), normalize their last names to match
- If co-buyer address is missing or incorrect, use buyer address if they appear to share it
- Addresses must start with a street number (e.g., "500 Ricky Street", not "Suu")
- Only correct values that are clearly visible in the OCR text
- Return null for fields that cannot be corrected with confidence

Return ONLY valid JSON matching the schema structure."""
    
    def _build_correction_prompt(
        self,
        schema: InstallmentAgreementSchema,
        ocr_result: OCRResult,
        issues: List[ValidationIssue]
    ) -> str:
        """Build prompt for AI correction."""
        prompt_parts = []
        
        prompt_parts.append("=== VALIDATION ISSUES DETECTED ===")
        for issue in issues:
            prompt_parts.append(f"- {issue.field}: {issue.issue_type} - {issue.description} (severity: {issue.severity})")
        
        prompt_parts.append("\n=== CURRENT EXTRACTED DATA ===")
        current_data = schema.to_json_dict()
        prompt_parts.append(json.dumps(current_data, indent=2, ensure_ascii=False))
        
        prompt_parts.append("\n=== OCR TEXT (for reference) ===")
        # Include relevant sections of OCR text
        ocr_text = ocr_result.full_text
        prompt_parts.append(ocr_text[:4000])  # First 4000 chars
        if len(ocr_text) > 4000:
            prompt_parts.append(f"\n[OCR text truncated. Total length: {len(ocr_text)} characters]")
        
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append("""
Please correct the extracted data based on the validation issues:

1. **Name Normalization:**
   - If buyer and co-buyer appear to be a couple, ensure they share the same last name
   - Fix OCR errors in names (e.g., "Hornberse" → "Hornberger")
   - Use the OCR text to determine the correct spelling

2. **Address Correction:**
   - Fix incomplete addresses (e.g., "Suu" → "500 Ricky Street")
   - Remove phone numbers that were incorrectly included in addresses
   - Ensure addresses start with a street number
   - If buyer and co-buyer share an address, ensure both have the same address
   - For seller address: Parse into street address, city, state, and ZIP code if full address is provided

3. **Seller Information:**
   - Extract seller name, address, city, state, ZIP code, and phone number
   - Parse seller address into components (city, state, zip_code) if full address is provided
   - Ensure seller phone number is properly formatted

4. **General Rules:**
   - Only correct fields where the correct value is clearly visible in OCR text
   - Do NOT guess or invent data
   - If unsure, keep the original value or use null

Return a JSON object with the corrected data using the same schema structure.
""")
        
        return "\n".join(prompt_parts)

