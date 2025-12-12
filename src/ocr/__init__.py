"""OCR module for extracting text from images using Google Cloud Vision API."""

from .vision_client import VisionOCRClient, OCRResult

__all__ = ['VisionOCRClient', 'OCRResult']

