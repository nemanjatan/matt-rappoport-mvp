"""Logging utilities for extraction pipeline debugging."""

import os
import logging
import json
from typing import Optional, Dict, Any
from functools import wraps

# Global debug mode flag
DEBUG_MODE = os.getenv("EXTRACTION_DEBUG", "false").lower() == "true"

# Logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(level: int = logging.INFO, debug_mode: bool = None) -> logging.Logger:
    """
    Set up logger for extraction pipeline.
    
    Args:
        level: Logging level (default: INFO)
        debug_mode: Override debug mode (default: from env var)
    
    Returns:
        Configured logger instance
    """
    global _logger, DEBUG_MODE
    
    if debug_mode is not None:
        DEBUG_MODE = debug_mode
    
    if _logger is None:
        _logger = logging.getLogger("extraction_pipeline")
        _logger.setLevel(logging.DEBUG if DEBUG_MODE else level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if DEBUG_MODE else level)
        
        # Create formatter
        if DEBUG_MODE:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        
        # Prevent duplicate logs
        _logger.propagate = False
    
    return _logger


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    if _logger is None:
        return setup_logger()
    return _logger


def log_ocr_result(logger: logging.Logger, ocr_result: Any, debug: bool = False):
    """
    Log OCR extraction results.
    
    Args:
        logger: Logger instance
        ocr_result: OCRResult object
        debug: If True, log full text
    """
    logger.info("=" * 60)
    logger.info("OCR EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Text length: {len(ocr_result.full_text)} characters")
    logger.info(f"Words extracted: {len(ocr_result.word_annotations)}")
    logger.info(f"Blocks extracted: {len(ocr_result.block_annotations)}")
    
    # Log confidence scores
    if ocr_result.confidence_scores:
        word_level = ocr_result.confidence_scores.get('word_level', {})
        block_level = ocr_result.confidence_scores.get('block_level', {})
        
        if word_level:
            logger.info(f"Word-level confidence: mean={word_level.get('mean', 0):.3f}, "
                       f"min={word_level.get('min', 0):.3f}, max={word_level.get('max', 0):.3f}")
        if block_level:
            logger.info(f"Block-level confidence: mean={block_level.get('mean', 0):.3f}, "
                       f"min={block_level.get('min', 0):.3f}, max={block_level.get('max', 0):.3f}")
    
    # Log warnings
    if ocr_result.warnings:
        logger.warning(f"OCR warnings: {ocr_result.warnings}")
    
    # Log raw text in debug mode
    if debug and DEBUG_MODE:
        logger.debug("=" * 60)
        logger.debug("RAW OCR TEXT:")
        logger.debug("=" * 60)
        logger.debug(ocr_result.full_text[:2000])  # First 2000 chars
        if len(ocr_result.full_text) > 2000:
            logger.debug(f"... (truncated, total length: {len(ocr_result.full_text)})")


def log_extraction_candidates(logger: logging.Logger, field_name: str, candidates: list):
    """
    Log extraction candidates for a field.
    
    Args:
        logger: Logger instance
        field_name: Name of the field
        candidates: List of FieldCandidate objects
    """
    if not DEBUG_MODE:
        return
    
    logger.debug(f"  Field: {field_name}")
    if candidates:
        logger.debug(f"    Found {len(candidates)} candidate(s):")
        for i, candidate in enumerate(candidates[:5], 1):  # Top 5
            logger.debug(f"      {i}. '{candidate.value}' "
                       f"(distance: {candidate.distance:.1f}, "
                       f"confidence: {candidate.confidence:.3f})")
    else:
        logger.debug(f"    No candidates found")


def log_field_extraction(logger: logging.Logger, field_name: str, value: Any, source: str = "deterministic"):
    """
    Log field extraction result.
    
    Args:
        logger: Logger instance
        field_name: Name of the field
        value: Extracted value
        source: Source of extraction (deterministic/openai)
    """
    if value is not None:
        logger.info(f"  ✓ {field_name:25s}: {value} ({source})")
    else:
        logger.warning(f"  ✗ {field_name:25s}: None ({source})")


def redact_sensitive_data(text: str, redact_api_keys: bool = True) -> str:
    """
    Redact sensitive data from logs.
    
    Args:
        text: Text to redact
        redact_api_keys: Whether to redact API keys
    
    Returns:
        Redacted text
    """
    if not redact_api_keys:
        return text
    
    # Redact OpenAI API keys (sk-...)
    import re
    text = re.sub(r'sk-[A-Za-z0-9]{32,}', 'sk-REDACTED', text)
    
    # Redact other common API key patterns
    text = re.sub(r'AIza[0-9A-Za-z_-]{35}', 'AIzaREDACTED', text)
    
    return text


def log_openai_request(logger: logging.Logger, prompt: str, model: str, redact: bool = True):
    """
    Log OpenAI request.
    
    Args:
        logger: Logger instance
        prompt: Prompt sent to OpenAI
        model: Model name
        redact: Whether to redact sensitive data
    """
    if not DEBUG_MODE:
        return
    
    logger.debug("=" * 60)
    logger.debug("OPENAI REQUEST")
    logger.debug("=" * 60)
    logger.debug(f"Model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    if redact:
        prompt = redact_sensitive_data(prompt)
    
    # Log prompt (truncated if very long)
    if len(prompt) > 2000:
        logger.debug(f"Prompt (first 2000 chars):\n{prompt[:2000]}...")
        logger.debug(f"... (truncated, total length: {len(prompt)})")
    else:
        logger.debug(f"Prompt:\n{prompt}")


def log_openai_response(logger: logging.Logger, response: str, redact: bool = True):
    """
    Log OpenAI response.
    
    Args:
        logger: Logger instance
        response: Response from OpenAI
        redact: Whether to redact sensitive data
    """
    if not DEBUG_MODE:
        return
    
    logger.debug("=" * 60)
    logger.debug("OPENAI RESPONSE")
    logger.debug("=" * 60)
    
    if redact:
        response = redact_sensitive_data(response)
    
    # Try to parse as JSON for pretty printing
    try:
        parsed = json.loads(response)
        logger.debug("Response (JSON):")
        logger.debug(json.dumps(parsed, indent=2, ensure_ascii=False))
    except:
        # Not JSON, log as text
        if len(response) > 2000:
            logger.debug(f"Response (first 2000 chars):\n{response[:2000]}...")
        else:
            logger.debug(f"Response:\n{response}")


def log_openai_usage(logger: logging.Logger, used: bool, reason: str = None):
    """
    Log whether OpenAI was used.
    
    Args:
        logger: Logger instance
        used: Whether OpenAI was used
        reason: Reason for using/not using OpenAI
    """
    logger.info("=" * 60)
    if used:
        logger.info("🤖 OPENAI ENHANCEMENT: ENABLED")
        if reason:
            logger.info(f"   Reason: {reason}")
    else:
        logger.info("📋 DETERMINISTIC EXTRACTION ONLY")
        if reason:
            logger.info(f"   Reason: {reason}")
    logger.info("=" * 60)

