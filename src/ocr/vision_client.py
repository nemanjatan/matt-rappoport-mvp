"""Google Cloud Vision API client for OCR text extraction."""

import os
import json
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image


class OCRResult:
    """Container for OCR extraction results."""
    
    def __init__(
        self,
        full_text: str,
        word_annotations: List[Dict[str, Any]],
        block_annotations: List[Dict[str, Any]],
        confidence_scores: Dict[str, float],
        raw_response: Dict[str, Any],
        warnings: List[str]
    ):
        """
        Initialize OCR result.
        
        Args:
            full_text: Complete extracted text
            word_annotations: List of word-level annotations with bounding boxes
            block_annotations: List of block-level annotations with bounding boxes
            confidence_scores: Dictionary with word-level and block-level confidence scores
            raw_response: Raw API response for debugging
            warnings: List of warnings from the API
        """
        self.full_text = full_text
        self.word_annotations = word_annotations
        self.block_annotations = block_annotations
        self.confidence_scores = confidence_scores
        self.raw_response = raw_response
        self.warnings = warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary."""
        return {
            'full_text': self.full_text,
            'word_annotations': self.word_annotations,
            'block_annotations': self.block_annotations,
            'confidence_scores': self.confidence_scores,
            'warnings': self.warnings,
            'raw_response': self.raw_response
        }
    
    def __repr__(self) -> str:
        return f"OCRResult(full_text_length={len(self.full_text)}, words={len(self.word_annotations)}, blocks={len(self.block_annotations)})"


class VisionOCRClient:
    """Client for Google Cloud Vision API OCR operations."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Vision OCR client.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
                            For Railway: Set GOOGLE_APPLICATION_CREDENTIALS as env var
                            pointing to file path, or Railway will handle it automatically.
        """
        # Set credentials path if provided
        # Railway will typically set GOOGLE_APPLICATION_CREDENTIALS as an env var
        # pointing to the credentials file path
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        # If no credentials_path provided, GOOGLE_APPLICATION_CREDENTIALS env var will be used
        # This works for both local development and Railway deployment
        
        self.client = vision.ImageAnnotatorClient()
    
    def extract_text(
        self,
        image_path: str,
        save_raw_output: bool = True,
        output_dir: Optional[str] = None
    ) -> OCRResult:
        """
        Extract text from an image using DOCUMENT_TEXT_DETECTION.
        
        Args:
            image_path: Path to image file (PNG or JPG)
            save_raw_output: Whether to save raw OCR output to file
            output_dir: Directory to save raw output (defaults to 'output' directory)
        
        Returns:
            OCRResult object containing extracted text, annotations, and metadata
        """
        # Validate image format
        image_ext = Path(image_path).suffix.lower()
        if image_ext not in ['.png', '.jpg', '.jpeg']:
            raise ValueError(f"Unsupported image format: {image_ext}. Use PNG or JPG.")
        
        # Read image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        # Create image object
        image = vision.Image(content=content)
        
        # Perform document text detection
        response = self.client.document_text_detection(image=image)
        
        # Check for errors
        if response.error.message:
            raise Exception(f"API Error: {response.error.message}")
        
        # Extract full text annotation
        full_text_annotation = response.full_text_annotation
        
        # Extract full text
        full_text = full_text_annotation.text if full_text_annotation else ""
        
        # Extract word-level annotations with bounding boxes and confidence
        word_annotations = self._extract_word_annotations(full_text_annotation)
        
        # Extract block-level annotations with bounding boxes and confidence
        block_annotations = self._extract_block_annotations(full_text_annotation)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            word_annotations, block_annotations
        )
        
        # Extract warnings
        warnings = []
        if hasattr(response, 'text_annotations') and response.text_annotations:
            # Check for any warnings in the response
            pass  # Google Vision API doesn't typically return warnings in this format
        
        # Prepare raw response for debugging
        raw_response = self._serialize_response(response)
        
        # Log OCR results
        try:
            from src.utils import get_logger, log_ocr_result
            logger = get_logger()
            ocr_result = OCRResult(
                full_text=full_text,
                word_annotations=word_annotations,
                block_annotations=block_annotations,
                confidence_scores=confidence_scores,
                raw_response=raw_response,
                warnings=warnings
            )
            log_ocr_result(logger, ocr_result, debug=True)
            return ocr_result
        except ImportError:
            # Logging not available, return without logging
            pass
        
        # Save raw output if requested
        if save_raw_output:
            self._save_raw_output(
                image_path, raw_response, output_dir
            )
        
        return OCRResult(
            full_text=full_text,
            word_annotations=word_annotations,
            block_annotations=block_annotations,
            confidence_scores=confidence_scores,
            raw_response=raw_response,
            warnings=warnings
        )
    
    def _extract_word_annotations(
        self, full_text_annotation: types.TextAnnotation
    ) -> List[Dict[str, Any]]:
        """Extract word-level annotations with bounding boxes and confidence."""
        word_annotations = []
        
        if not full_text_annotation or not full_text_annotation.pages:
            return word_annotations
        
        for page in full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        # Extract word text
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        
                        # Extract bounding box
                        vertices = [
                            {'x': vertex.x, 'y': vertex.y}
                            for vertex in word.bounding_box.vertices
                        ]
                        
                        # Extract confidence score
                        confidence = word.confidence if hasattr(word, 'confidence') else None
                        
                        word_annotations.append({
                            'text': word_text,
                            'bounding_box': vertices,
                            'confidence': confidence
                        })
        
        return word_annotations
    
    def _extract_block_annotations(
        self, full_text_annotation: types.TextAnnotation
    ) -> List[Dict[str, Any]]:
        """Extract block-level annotations with bounding boxes and confidence."""
        block_annotations = []
        
        if not full_text_annotation or not full_text_annotation.pages:
            return block_annotations
        
        for page in full_text_annotation.pages:
            for block in page.blocks:
                # Extract block text
                block_text = self._extract_block_text(block)
                
                # Extract bounding box
                vertices = [
                    {'x': vertex.x, 'y': vertex.y}
                    for vertex in block.bounding_box.vertices
                ]
                
                # Extract confidence score (average of word confidences in block)
                block_confidence = self._calculate_block_confidence(block)
                
                block_annotations.append({
                    'text': block_text,
                    'bounding_box': vertices,
                    'confidence': block_confidence,
                    'block_type': self._get_block_type(block)
                })
        
        return block_annotations
    
    def _extract_block_text(self, block: types.Block) -> str:
        """Extract text from a block."""
        text_parts = []
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                word_text = ''.join([
                    symbol.text for symbol in word.symbols
                ])
                text_parts.append(word_text)
        return ' '.join(text_parts)
    
    def _calculate_block_confidence(self, block: types.Block) -> Optional[float]:
        """Calculate average confidence score for a block."""
        confidences = []
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                if hasattr(word, 'confidence') and word.confidence is not None:
                    confidences.append(word.confidence)
        
        if confidences:
            return sum(confidences) / len(confidences)
        return None
    
    def _get_block_type(self, block: types.Block) -> str:
        """Get block type as string."""
        block_type_map = {
            types.Block.BlockType.UNKNOWN: 'UNKNOWN',
            types.Block.BlockType.TEXT: 'TEXT',
            types.Block.BlockType.TABLE: 'TABLE',
            types.Block.BlockType.PICTURE: 'PICTURE',
            types.Block.BlockType.RULER: 'RULER',
            types.Block.BlockType.BARCODE: 'BARCODE'
        }
        return block_type_map.get(block.block_type, 'UNKNOWN')
    
    def _calculate_confidence_scores(
        self,
        word_annotations: List[Dict[str, Any]],
        block_annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate confidence scores."""
        scores = {}
        
        # Word-level confidence
        word_confidences = [
            w['confidence'] for w in word_annotations
            if w['confidence'] is not None
        ]
        if word_confidences:
            scores['word_level'] = {
                'mean': sum(word_confidences) / len(word_confidences),
                'min': min(word_confidences),
                'max': max(word_confidences)
            }
        
        # Block-level confidence
        block_confidences = [
            b['confidence'] for b in block_annotations
            if b['confidence'] is not None
        ]
        if block_confidences:
            scores['block_level'] = {
                'mean': sum(block_confidences) / len(block_confidences),
                'min': min(block_confidences),
                'max': max(block_confidences)
            }
        
        return scores
    
    def _serialize_response(self, response) -> Dict[str, Any]:
        """Serialize API response to dictionary for debugging."""
        # Convert protobuf message to dict
        try:
            from google.protobuf.json_format import MessageToDict
            # Serialize the full response
            response_dict = MessageToDict(response._pb)
            return response_dict
        except (ImportError, AttributeError) as e:
            # Fallback: return basic info if protobuf json_format not available
            # or if response structure is different
            return {
                'text_annotations_count': len(response.text_annotations) if response.text_annotations else 0,
                'has_full_text_annotation': response.full_text_annotation is not None,
                'serialization_error': str(e)
            }
    
    def _save_raw_output(
        self,
        image_path: str,
        raw_response: Dict[str, Any],
        output_dir: Optional[str]
    ) -> None:
        """Save raw OCR output to JSON file."""
        if output_dir is None:
            output_dir = 'output'
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{image_name}_ocr_raw.json"
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_response, f, indent=2, ensure_ascii=False)
        
        print(f"Raw OCR output saved to: {output_path}")

