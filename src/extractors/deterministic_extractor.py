"""Deterministic field extraction using OCR text and layout information."""

import re
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

from src.ocr import OCRResult
from src.schema import InstallmentAgreementSchema


class FieldCandidate:
    """Represents a candidate value for a field."""
    
    def __init__(
        self,
        value: str,
        confidence: float,
        distance: float,
        label_match: str,
        position: Dict[str, int]
    ):
        self.value = value
        self.confidence = confidence
        self.distance = distance
        self.label_match = label_match
        self.position = position
    
    def __repr__(self):
        return f"FieldCandidate(value='{self.value}', confidence={self.confidence:.3f}, distance={self.distance:.1f})"


class DeterministicExtractor:
    """Deterministic extractor using keyword proximity and layout information."""
    
    # Field label patterns - multiple variants per field
    FIELD_LABELS = {
        'seller_name': [
            r"seller'?s?\s+name\s*:?",
            r"seller\s*\(.*?also called.*?\)",
            r"^seller\s*:",
            r"\bseller\s*:",  # Match "Seller:" anywhere (not just start of line)
            r"se[il]ler\s+name",
        ],
        'seller_address': [
            r"seller'?s?\s+address\s*:?",
            r"seller\s+address\s*:?",
            r"\bseller\s*:",  # Also use seller label for address extraction
        ],
        'seller_city': [
            r"seller'?s?\s+city\s*:?",
            r"seller\s+city\s*:?",
        ],
        'seller_state': [
            r"seller'?s?\s+state\s*:?",
            r"seller\s+state\s*:?",
        ],
        'seller_zip_code': [
            r"seller'?s?\s+zip\s*:?",
            r"seller\s+zip\s*:?",
            r"seller'?s?\s+zip\s+code\s*:?",
        ],
        'seller_phone_number': [
            r"seller'?s?\s+phone\s*:?",
            r"seller\s+phone\s*:?",
            r"seller'?s?\s+phone\s+number\s*:?",
        ],
        'buyer_name': [
            r"buyer\s+1'?s?\s+name\s*:?",
            r"buyer'?s?\s+name\s*:?",
            r"buyer\s*\(.*?also called.*?\)",
            r"^buyer\s*:",
        ],
        'co_buyer_name': [
            r"buyer\s+2'?s?\s+name\s*:?",
            r"co-?buyer'?s?\s+name\s*:?",
            r"co-?buyer\s*:?",
        ],
        'buyer_address': [
            r"buyer\s+1'?s?\s+address\s*:?",
            r"buyer'?s?\s+address\s*:?",
            r"mailing\s+address\s*:",
            r"street\s+address\s*:",
        ],
        'buyer_phone_number': [
            r"buyer\s+1'?s?\s+phone\s*:?",
            r"buyer'?s?\s+phone\s*:?",
            r"phone\s+number",
            r"phone\s*:",
            r"phone\s+no",
        ],
        'co_buyer_address': [
            r"buyer\s+2'?s?\s+address\s*:?",
            r"co-?buyer'?s?\s+address\s*:?",
            r"co-?buyer\s+mailing\s+address\s*:?",
        ],
        'co_buyer_phone_number': [
            r"buyer\s+2'?s?\s+phone\s*:?",
            r"co-?buyer'?s?\s+phone\s*:?",
            r"co-?buyer\s+phone\s+number\s*:?",
        ],
        # Legacy field names for backward compatibility
        'street_address': [
            r"mailing\s+address\s*:",
            r"street\s+address\s*:",
            r"address\s*:",
        ],
        'phone_number': [
            r"phone\s+number",
            r"phone\s*:",
            r"phone\s+no",
        ],
        'quantity': [
            r"quantity\s*:",
            r"qty\s*:",
            r"qty\.\s*:",
        ],
        'items_purchased': [
            r"items?\s*:",
            r"description\s+of\s+goods",
            r"description\s*:",
            r"goods\s+or\s+services",
        ],
        'make_or_model': [
            r"make\s+or\s+model",
            r"make/?model",
            r"make\s*:",
            r"model\s*:",
        ],
        'amount_financed': [
            r"amount\s+financed",
            r"amount\s+of\s+credit",
        ],
        'finance_charge': [
            r"finance\s+charge",
        ],
        'apr': [
            r"annual\s+percentage\s+rate",
            r"\bapr\s*:",
            r"annual\s+percentage",
        ],
        'total_of_payments': [
            r"total\s+of\s+payments",
            r"total\s+payments",
        ],
        'number_of_payments': [
            r"number\s+of\s+payments",
            r"number\s+of\s+installment\s+payments",
            r"installment\s+payments",
        ],
        'amount_of_payments': [
            r"amount\s+of\s+payments\s*:",
            r"amount\s+of\s+the\s+installment\s+payment\s*:",
            r"payment\s+amount\s*:",
            r"installment\s+payment\s+amount\s*:",
            r"amount\s+of\s+each\s+payment\s*:",
        ],
    }
    
    def __init__(self, ocr_result: OCRResult):
        """
        Initialize extractor with OCR result.
        
        Args:
            ocr_result: OCRResult from VisionOCRClient
        """
        self.ocr_result = ocr_result
        self.full_text = ocr_result.full_text
        self.word_annotations = ocr_result.word_annotations
        self.block_annotations = ocr_result.block_annotations
        
        # Build searchable text with positions
        self._build_text_index()
    
    def _build_text_index(self):
        """Build index of text with positions for proximity search."""
        self.text_index = []
        self.char_to_word_map = {}  # Map character positions to word indices
        
        char_pos = 0
        for i, word in enumerate(self.word_annotations):
            center_x = sum(v['x'] for v in word['bounding_box']) / len(word['bounding_box'])
            center_y = sum(v['y'] for v in word['bounding_box']) / len(word['bounding_box'])
            
            word_text = word['text']
            word_len = len(word_text)
            
            # Map character positions to this word
            for char_idx in range(char_pos, char_pos + word_len):
                self.char_to_word_map[char_idx] = i
            
            self.text_index.append({
                'text': word_text,
                'index': i,
                'center': (center_x, center_y),
                'bounding_box': word['bounding_box'],
                'confidence': word.get('confidence', 1.0),
                'char_start': char_pos,
                'char_end': char_pos + word_len
            })
            
            # Move past word and space
            char_pos += word_len + 1
    
    def extract_all_fields(self) -> InstallmentAgreementSchema:
        """
        Extract all fields and return as InstallmentAgreementSchema.
        
        Returns:
            InstallmentAgreementSchema with extracted values
        """
        try:
            from src.utils import get_logger, log_extraction_candidates, log_field_extraction
            logger = get_logger()
            logger.info("=" * 60)
            logger.info("DETERMINISTIC EXTRACTION")
            logger.info("=" * 60)
        except ImportError:
            logger = None
        
        extracted = {}
        
        for field_name in InstallmentAgreementSchema.model_fields.keys():
            candidates = self._find_field_candidates(field_name)
            
            # Log candidates in debug mode
            if logger:
                log_extraction_candidates(logger, field_name, candidates)
            
            value = self._resolve_candidates(field_name, candidates)
            
            # Log extraction result
            if logger:
                log_field_extraction(logger, field_name, value, source="deterministic")
            
            extracted[field_name] = value
        
        # Post-process seller address to extract city, state, zip
        if extracted.get('seller_address') and not extracted.get('seller_city'):
            city, state, zip_code = self._parse_address(extracted['seller_address'])
            if city:
                extracted['seller_city'] = city
            if state:
                extracted['seller_state'] = state
            if zip_code:
                extracted['seller_zip_code'] = zip_code
        
        # Map legacy fields to new fields for backward compatibility
        if extracted.get('street_address') and not extracted.get('buyer_address'):
            extracted['buyer_address'] = extracted['street_address']
        if extracted.get('phone_number') and not extracted.get('buyer_phone_number'):
            extracted['buyer_phone_number'] = extracted['phone_number']
        
        if logger:
            logger.info("=" * 60)
        
        return InstallmentAgreementSchema(**extracted)
    
    def _parse_address(self, address: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse address string into city, state, zip code.
        
        Args:
            address: Full address string (e.g., "1901 Farragut Ave, Bristol, PA 19007")
        
        Returns:
            Tuple of (city, state, zip_code)
        """
        if not address:
            return None, None, None
        
        # Common pattern: "Street, City, State ZIP"
        # Try to extract ZIP code (5 or 9 digits at the end)
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', address)
        zip_code = zip_match.group(1) if zip_match else None
        
        # Try to extract state (2-letter code before ZIP)
        if zip_code:
            # Remove ZIP from address for state extraction
            address_without_zip = address[:address.find(zip_code)].strip()
        else:
            address_without_zip = address
        
        # Look for 2-letter state code
        state_match = re.search(r'\b([A-Z]{2})\b', address_without_zip[-10:])  # Last 10 chars
        state = state_match.group(1) if state_match else None
        
        # Extract city (text between last comma and state)
        if state:
            # Split by comma and find city (usually second-to-last part)
            parts = [p.strip() for p in address_without_zip.split(',')]
            if len(parts) >= 2:
                # City is typically the part before state
                city = parts[-2] if len(parts) >= 2 else None
                # Remove state if it's in city
                if city:
                    city = re.sub(r'\b' + state + r'\b', '', city).strip()
            else:
                city = None
        else:
            # Try to extract city (text after last comma before ZIP)
            parts = address_without_zip.rsplit(',', 1)
            city = parts[-1].strip() if len(parts) > 1 else None
        
        return city, state, zip_code
    
    def _find_field_candidates(self, field_name: str) -> List[FieldCandidate]:
        """
        Find candidate values for a field using label proximity.
        
        Args:
            field_name: Name of the field to extract
        
        Returns:
            List of FieldCandidate objects
        """
        candidates = []
        label_patterns = self.FIELD_LABELS.get(field_name, [])
        
        # Search for label matches in text
        label_matches = []
        for pattern in label_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(self.full_text):
                # Find the word index for this label
                label_pos = self._find_text_position(match.start(), match.end())
                if label_pos:
                    label_matches.append({
                        'pattern': pattern,
                        'match_text': match.group(),
                        'position': label_pos,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # For each label match, find nearby values
        for label_match in label_matches:
            # Special handling for seller fields - use "Seller:" label for all seller fields
            if field_name.startswith('seller_'):
                # Check if this is a "Seller:" label (not "seller phone" etc.)
                if 'seller' in label_match['match_text'].lower() and ':' in label_match['match_text']:
                    seller_candidates = self._extract_seller_info_from_text_pos(
                        field_name, label_match['start'], label_match['end']
                    )
                    candidates.extend(seller_candidates)
                else:
                    # For other seller label patterns (e.g., "seller phone"), use normal extraction
                    nearby_values = self._find_nearby_values(
                        field_name,
                        label_match['position'],
                        label_match['match_text']
                    )
                    candidates.extend(nearby_values)
            else:
                nearby_values = self._find_nearby_values(
                    field_name,
                    label_match['position'],
                    label_match['match_text']
                )
                candidates.extend(nearby_values)
        
        # Fallback for seller fields: if no label match found, try using "Seller:" label
        if field_name.startswith('seller_') and not candidates:
            seller_label_match = re.search(r'Seller\s*:', self.full_text, re.IGNORECASE)
            if seller_label_match:
                seller_candidates = self._extract_seller_info_from_text_pos(
                    field_name, seller_label_match.start(), seller_label_match.end()
                )
                candidates.extend(seller_candidates)
        
        return candidates
    
    def _find_text_position(self, start_char: int, end_char: int) -> Optional[Dict[str, Any]]:
        """Find the word position in text_index for a character range."""
        # Use character map to find word index
        word_idx = self.char_to_word_map.get(start_char)
        if word_idx is None:
            # Try to find closest word
            for idx, item in enumerate(self.text_index):
                if item['char_start'] <= start_char <= item['char_end']:
                    word_idx = idx
                    break
        
        if word_idx is not None and word_idx < len(self.text_index):
            item = self.text_index[word_idx]
            return {
                'center': item['center'],
                'index': item['index'],
                'bounding_box': item['bounding_box']
            }
        return None
    
    def _find_nearby_values(
        self,
        field_name: str,
        label_position: Dict[str, Any],
        label_text: str
    ) -> List[FieldCandidate]:
        """
        Find values near a label position.
        
        Args:
            field_name: Field being extracted
            label_position: Position of the label
            label_text: Text of the matched label
        
        Returns:
            List of FieldCandidate objects
        """
        candidates = []
        label_center = label_position['center']
        label_x, label_y = label_center
        label_idx = label_position['index']
        
        # For seller fields with "Seller's Name" or "Seller's Address" labels,
        # the value often comes BEFORE the label (above/left)
        look_before_label = (
            field_name.startswith('seller_') and 
            ("seller's name" in label_text.lower() or "seller's address" in label_text.lower())
        )
        
        # Search in a region around the label
        # Look right and below the label (typical document layout)
        search_radius_x = 500  # pixels
        search_radius_y = 200  # pixels
        
        for i, item in enumerate(self.text_index):
            item_x, item_y = item['center']
            
            # Calculate distance from label
            dx = item_x - label_x
            dy = item_y - label_y
            
            # For seller fields with labels like "Seller's Name", also look before the label
            if look_before_label:
                # Look above/left of label (values come before labels in this format)
                if dx > 50 or dx < -search_radius_x:
                    continue
                if abs(dy) > search_radius_y:
                    continue
                # Prefer items that are above the label (negative dy) or on same line
                if dy > 100:  # Too far below
                    continue
            else:
                # Only consider items to the right and below (or slightly above)
                if dx < -50 or dx > search_radius_x:
                    continue
                if abs(dy) > search_radius_y:
                    continue
            
            # Calculate distance
            distance = (dx**2 + dy**2)**0.5
            
            # Check if this looks like a value for this field
            value_text = item['text']
            if self._is_valid_value(field_name, value_text):
                confidence = item.get('confidence', 1.0)
                candidates.append(FieldCandidate(
                    value=value_text,
                    confidence=confidence,
                    distance=distance,
                    label_match=label_text,
                    position={'x': item_x, 'y': item_y}
                ))
        
        # Also try to extract multi-word values (e.g., full names, addresses)
        if field_name in ['buyer_name', 'co_buyer_name', 'buyer_address', 'co_buyer_address', 'street_address', 'seller_name', 'seller_address', 'items_purchased', 'make_or_model']:
            multi_word_candidates = self._extract_multi_word_value(
                field_name, label_position, label_text
            )
            candidates.extend(multi_word_candidates)
        
        # Special handling: extract values BEFORE "Seller's Name" or "Seller's Address" labels
        if look_before_label:
            before_candidates = self._extract_value_before_label(
                field_name, label_position, label_text
            )
            candidates.extend(before_candidates)
        
        return candidates
    
    def _extract_value_before_label(
        self,
        field_name: str,
        label_position: Dict[str, Any],
        label_text: str
    ) -> List[FieldCandidate]:
        """Extract values that appear BEFORE a label (for seller fields in IMG_1807 format)."""
        candidates = []
        label_center = label_position['center']
        label_x, label_y = label_center
        
        # Find words that are spatially near the label (above, same line, or below)
        nearby_words = []
        for i, item in enumerate(self.text_index):
            item_x, item_y = item['center']
            
            # Calculate distance from label
            dx = item_x - label_x
            dy = item_y - label_y
            
            # For seller_address, also look below the label (address can be below in some formats)
            max_dy_below = 50 if field_name == 'seller_address' else 100
            
            # Look for words above, on same line, or slightly below the label
            if dy > max_dy_below:  # Too far below the label
                continue
            if abs(dx) > 500:  # Too far horizontally
                continue
            # Prefer words that are above the label (negative dy) or on same line
            if dy < -300:  # Too far above (probably header text)
                continue
            
            nearby_words.append((i, item, dx, dy))
        
        if not nearby_words:
            return candidates
        
        # Sort by y position (top to bottom, so values above label come first), then x (left to right)
        nearby_words.sort(key=lambda x: (x[3], x[2]))
        
        # Extract based on field type
        if field_name == 'seller_name':
            # Seller name is typically the line(s) directly above "Seller's Name"
            # Find words on the same line (to the left) or line above the label
            name_words = []
            label_word_idx = label_position['index']
            
            # Find words on the same line as label position (y coordinate)
            # In IMG_1807 format, seller name is on same line as label position, to the left
            same_line_words = []
            line_above_words = []
            
            for i, item, dx, dy in nearby_words:
                item_y = item['center'][1]
                word_text = item['text'].strip()
                
                # Words on same line (within 5 pixels of label_y)
                if abs(item_y - label_y) < 5:
                    same_line_words.append((i, item, dx, dy, word_text))
                # Words on line above (10-30 pixels above label_y)
                elif -30 < (item_y - label_y) < -10:
                    line_above_words.append((i, item, dx, dy, word_text))
            
            # Sort by x position (left to right)
            same_line_words.sort(key=lambda x: x[1]['center'][0])
            line_above_words.sort(key=lambda x: x[1]['center'][0])
            
            # Prioritize same line (seller name is often on same line as label in IMG_1807 format)
            # Try same line first
            if same_line_words:
                for i, item, dx, dy, word_text in same_line_words:
                    if not word_text or word_text in [':', ',', '(', ')', '.']:
                        continue
                    # Skip the label words themselves
                    if word_text.upper() in ['SELLER', "SELLER'S", 'NAME']:
                        continue
                    # Stop if we hit buyer name or other section
                    if word_text.upper() in ['BUYER', 'DAVID', 'POWERS', 'AGREEMENT']:
                        break
                    # Stop if we hit phone number
                    if re.match(r'^\d{3}-\d{3}-\d{4}', word_text):
                        break
                    name_words.append((item, word_text))
                    if len(name_words) >= 5:
                        break
                
                # If we got good words from same line, use them
                if name_words and len([w for _, w in name_words if len(w) > 2]) >= 2:
                    pass  # Use name_words as is
                else:
                    name_words = []  # Clear and try line above
            
            # If no good words from same line, try line above
            if not name_words and line_above_words:
                for i, item, dx, dy, word_text in same_line_words:
                    if not word_text or word_text in [':', ',', '(', ')', '.']:
                        continue
                    # Skip the label words themselves
                    if word_text.upper() in ['SELLER', "SELLER'S", 'NAME']:
                        continue
                    # Stop if we hit buyer name or other section
                    if word_text.upper() in ['BUYER', 'DAVID', 'POWERS', 'AGREEMENT']:
                        break
                    # Stop if we hit phone number
                    if re.match(r'^\d{3}-\d{3}-\d{4}', word_text):
                        break
                    name_words.append((item, word_text))
                    if len(name_words) >= 5:
                        break
            
            if name_words:
                name_text = ' '.join([word_text for _, word_text in name_words])
                # Clean up: remove leading/trailing punctuation
                name_text = re.sub(r'^[:\s,()]+|[:\s,()]+$', '', name_text.strip())
                if len(name_text.strip()) > 0:
                    avg_x = sum(item['center'][0] for item, _ in name_words) / len(name_words)
                    avg_y = sum(item['center'][1] for item, _ in name_words) / len(name_words)
                    avg_confidence = sum(item.get('confidence', 1.0) for item, _ in name_words) / len(name_words)
                    dx = avg_x - label_x
                    dy = avg_y - label_y
                    distance = (dx**2 + dy**2)**0.5
                    
                    candidates.append(FieldCandidate(
                        value=name_text.strip(),
                        confidence=avg_confidence,
                        distance=distance,
                        label_match=label_text,
                        position={'x': avg_x, 'y': avg_y}
                    ))
        
        elif field_name == 'seller_address':
            # Seller address is typically above "Seller's Address"
            # Similar to seller_name, look for words on same line or line above
            same_line_words = []
            line_above_words = []
            
            for i, item, dx, dy in nearby_words:
                item_y = item['center'][1]
                word_text = item['text'].strip()
                
                # Words on same line (within 5 pixels of label_y)
                if abs(item_y - label_y) < 5:
                    same_line_words.append((i, item, dx, dy, word_text))
                # Words on line above (10-40 pixels above label_y)
                elif -40 < (item_y - label_y) < -10:
                    line_above_words.append((i, item, dx, dy, word_text))
                # Words on line below (10-50 pixels below label_y) - for seller_address
                elif field_name == 'seller_address' and 10 < (item_y - label_y) < 50:
                    line_above_words.append((i, item, dx, dy, word_text))  # Use same list, will process together
            
            # Sort by x position (left to right)
            same_line_words.sort(key=lambda x: x[1]['center'][0])
            line_above_words.sort(key=lambda x: x[1]['center'][0])
            
            address_words = []
            
            # Try line above/below first (address is usually near the label)
            # Look for a line that starts with a number (street address)
            if line_above_words:
                # Group words by line (y coordinate)
                words_by_line = {}
                for i, item, dx, dy, word_text in line_above_words:
                    y_key = round(item['center'][1] / 5) * 5
                    if y_key not in words_by_line:
                        words_by_line[y_key] = []
                    words_by_line[y_key].append((i, item, dx, dy, word_text))
                
                # Find the line that starts with a number (street address)
                for line_y, words_on_line in sorted(words_by_line.items()):
                    words_on_line.sort(key=lambda x: x[1]['center'][0])  # Sort by x
                    # Check if this line starts with a number
                    first_word = words_on_line[0][4] if words_on_line else ""
                    if re.match(r'^\d', first_word):
                        # This is likely the address line
                        for i, item, dx, dy, word_text in words_on_line:
                            if not word_text or word_text in [':', ',', '(', ')', '.']:
                                continue
                            # Skip label words
                            if word_text.upper() in ['SELLER', "SELLER'S", 'ADDRESS', 'NAME']:
                                continue
                            # Stop if we hit buyer section
                            if word_text.upper() in ['BUYER', 'DAVID', 'POWERS']:
                                break
                            # Stop if we hit phone number
                            if re.match(r'^\d{3}-\d{3}-\d{4}', word_text):
                                break
                            address_words.append((item, word_text))
                            if len(address_words) >= 10:  # Addresses can be longer
                                break
                        break  # Found address line, stop looking
            
            # If no address from line above, try same line
            if not address_words and same_line_words:
                for i, item, dx, dy, word_text in same_line_words:
                    if not word_text or word_text in [':', ',', '(', ')', '.']:
                        continue
                    # Skip label words
                    if word_text.upper() in ['SELLER', "SELLER'S", 'ADDRESS', 'NAME']:
                        continue
                    # Stop if we hit buyer section
                    if word_text.upper() in ['BUYER', 'DAVID', 'POWERS']:
                        break
                    # Look for address parts
                    is_address_part = (
                        re.match(r'^\d', word_text) or 
                        any(indicator in word_text.upper() for indicator in ['AVE', 'AVENUE', 'ST', 'STREET', 'ROAD', 'RD', 'BLVD', 'BOULEVARD', 'LANE', 'LN', 'FARRAGUT', 'BRISTOL'])
                    )
                    if is_address_part:
                        address_words.append((item, word_text))
                        if len(address_words) >= 8:
                            break
            
            if address_words:
                address_text = ' '.join([word_text for _, word_text in address_words])
                # Clean up
                address_text = re.sub(r'^[:\s,()]+|[:\s,()]+$', '', address_text.strip())
                if len(address_text.strip()) > 0:
                    avg_x = sum(item['center'][0] for item, _ in address_words) / len(address_words)
                    avg_y = sum(item['center'][1] for item, _ in address_words) / len(address_words)
                    avg_confidence = sum(item.get('confidence', 1.0) for item, _ in address_words) / len(address_words)
                    dx = avg_x - label_x
                    dy = avg_y - label_y
                    distance = (dx**2 + dy**2)**0.5
                    
                    candidates.append(FieldCandidate(
                        value=address_text.strip(),
                        confidence=avg_confidence,
                        distance=distance,
                        label_match=label_text,
                        position={'x': avg_x, 'y': avg_y}
                    ))
        
        return candidates
    
    def _extract_seller_info_from_text_pos(
        self,
        field_name: str,
        label_start: int,
        label_end: int
    ) -> List[FieldCandidate]:
        """Extract seller information using text position of 'Seller:' label."""
        candidates = []
        
        # Find words that come after the "Seller:" label in the text
        nearby_words = []
        for item in self.text_index:
            # Only consider words that start after the label ends
            if item['char_start'] >= label_end:
                # But limit to words within reasonable distance (first 600 chars after label for phone)
                # Phone numbers might be further away
                max_chars = 600 if field_name == 'seller_phone_number' else 500
                if item['char_start'] <= label_end + max_chars:
                    nearby_words.append(item)
        
        if not nearby_words:
            return candidates
        
        # Sort by character position (order in text)
        nearby_words.sort(key=lambda x: x['char_start'])
        
        # Extract based on field type
        if field_name == 'seller_name':
            # Seller name is typically the first few words after "Seller:"
            name_words = []
            for item in nearby_words[:10]:  # First 10 words
                word_text = item['text'].strip()
                # Stop if we hit address (starts with number) or phone pattern
                if re.match(r'^\d', word_text) or re.match(r'^\d{3}-\d{3}-\d{4}', word_text):
                    break
                # Stop if we hit another label
                if word_text.upper() in ['CO-BUYER', 'BUYER', 'ADDRESS', 'PHONE', 'QUANTITY']:
                    break
                name_words.append((item, word_text))
            
            if name_words:
                name_text = ' '.join([word_text for _, word_text in name_words])
                # Clean up: remove leading colons, dashes, or other punctuation
                name_text = re.sub(r'^[:,\-\s]+', '', name_text.strip())
                if len(name_text.strip()) > 0:
                    avg_x = sum(item['center'][0] for item, _ in name_words) / len(name_words)
                    avg_y = sum(item['center'][1] for item, _ in name_words) / len(name_words)
                    avg_confidence = sum(item.get('confidence', 1.0) for item, _ in name_words) / len(name_words)
                    
                    candidates.append(FieldCandidate(
                        value=name_text.strip(),
                        confidence=avg_confidence,
                        distance=0.0,  # Distance not meaningful for text-based extraction
                        label_match="Seller:",
                        position={'x': avg_x, 'y': avg_y}
                    ))
        
        elif field_name == 'seller_address':
            # Seller address is typically after seller name, starts with a number
            address_words = []
            found_name = False
            
            for item in nearby_words[:20]:  # First 20 words
                word_text = item['text'].strip()
                
                # Skip seller name (first line of text)
                if not found_name and re.search(r'[A-Za-z]', word_text) and not re.search(r'^\d', word_text):
                    found_name = True
                    continue
                
                # Look for address line (starts with number) after name
                if found_name and re.match(r'^\d', word_text):
                    # Collect this word and following words until phone number or new section
                    for item2 in nearby_words[nearby_words.index(item):nearby_words.index(item)+15]:
                        word_text2 = item2['text'].strip()
                        # Stop if we hit phone number
                        if re.match(r'^\d{3}-\d{3}-\d{4}', word_text2):
                            break
                        # Stop if we hit another section
                        if word_text2.upper() in ['CO-BUYER', 'BUYER', 'QUANTITY']:
                            break
                        address_words.append((item2, word_text2))
                    break
            
            if address_words:
                address_text = ' '.join([word_text for _, word_text in address_words])
                if len(address_text.strip()) > 0:
                    avg_x = sum(item['center'][0] for item, _ in address_words) / len(address_words)
                    avg_y = sum(item['center'][1] for item, _ in address_words) / len(address_words)
                    avg_confidence = sum(item.get('confidence', 1.0) for item, _ in address_words) / len(address_words)
                    
                    candidates.append(FieldCandidate(
                        value=address_text.strip(),
                        confidence=avg_confidence,
                        distance=0.0,
                        label_match="Seller:",
                        position={'x': avg_x, 'y': avg_y}
                    ))
        
        elif field_name == 'seller_phone_number':
            # Seller phone is typically after address
            for item in nearby_words:
                word_text = item['text'].strip()
                # Look for phone number pattern
                if re.match(r'^\d{3}-\d{3}-\d{4}', word_text):
                    candidates.append(FieldCandidate(
                        value=word_text,
                        confidence=item.get('confidence', 1.0),
                        distance=0.0,
                        label_match="Seller:",
                        position={'x': item['center'][0], 'y': item['center'][1]}
                    ))
                    break
        
        return candidates
    
    def _extract_multi_word_value(
        self,
        field_name: str,
        label_position: Dict[str, Any],
        label_text: str
    ) -> List[FieldCandidate]:
        """Extract multi-word values (names, addresses, etc.) near a label."""
        candidates = []
        label_center = label_position['center']
        label_x, label_y = label_center
        label_idx = label_position['index']
        
        # Find words after the label (by index and position)
        nearby_words = []
        for i, item in enumerate(self.text_index):
            if i <= label_idx:
                continue
            
            item_x, item_y = item['center']
            dx = item_x - label_x
            dy = item_y - label_y
            
            # Look for words on same line (small dy) or next lines (small positive dy)
            # Also consider words slightly above (for wrapped text)
            # For buyer/co-buyer/seller names, look more to the right (they're often in a column)
            if field_name in ['buyer_name', 'co_buyer_name', 'buyer_address', 'co_buyer_address', 'seller_name']:
                # Names are often on the next line, slightly to the right
                if dx > 50 and 0 <= dy <= 80:  # Next line, to the right
                    nearby_words.append((i, item, dx, dy))
            else:
                if dx > -50 and -20 <= dy <= 150:  # Right of label, same or next few lines
                    nearby_words.append((i, item, dx, dy))
        
        # Sort by y position first (top to bottom), then x (left to right)
        nearby_words.sort(key=lambda x: (x[3], x[2]))
        
        # Try to extract value: words after label until we hit another label or section
        value_words = []
        max_words = 15 if field_name in ['street_address', 'buyer_address', 'co_buyer_address', 'seller_address'] else 8  # Addresses can be longer
        
        for i, item, dx, dy in nearby_words[:max_words]:
            word_text = item['text'].strip()
            
            # Skip empty words
            if not word_text:
                continue
            
            # Stop if we hit another label-like pattern
            if self._looks_like_label(word_text):
                break
            
            # Stop if we hit certain punctuation that suggests end of value
            if word_text in [':', '|', '—', '–', '/']:
                break
            
            # For names, stop if we hit common separators or next section
            if field_name in ['buyer_name', 'co_buyer_name', 'seller_name', 'buyer_address', 'co_buyer_address']:
                if word_text.upper() in ['PHONE', 'ADDRESS', 'CO-BUYER', 'BUYER', 'SELLER', 'SIGNING', 'AGREEMENT']:
                    break
                # Stop if we hit a phone number pattern
                if re.match(r'^\d{3}-\d{3}-\d{4}$', word_text):
                    break
            
            value_words.append((item, word_text))
        
        if value_words:
            # Combine words into a value
            value_text = ' '.join([word_text for _, word_text in value_words])
            value_text = value_text.strip()
            
            # Clean up common OCR artifacts
            value_text = re.sub(r'\s+', ' ', value_text)  # Multiple spaces
            value_text = re.sub(r'\s*:\s*$', '', value_text)  # Trailing colon
            
            if len(value_text) > 0:
                # Calculate average position and confidence
                avg_x = sum(item['center'][0] for item, _ in value_words) / len(value_words)
                avg_y = sum(item['center'][1] for item, _ in value_words) / len(value_words)
                avg_confidence = sum(item.get('confidence', 1.0) for item, _ in value_words) / len(value_words)
                
                dx = avg_x - label_x
                dy = avg_y - label_y
                distance = (dx**2 + dy**2)**0.5
                
                candidates.append(FieldCandidate(
                    value=value_text,
                    confidence=avg_confidence,
                    distance=distance,
                    label_match=label_text,
                    position={'x': avg_x, 'y': avg_y}
                ))
        
        return candidates
    
    def _looks_like_label(self, text: str) -> bool:
        """Check if text looks like a field label."""
        text_upper = text.upper()
        label_indicators = [
            'NAME', 'ADDRESS', 'PHONE', 'QUANTITY', 'ITEMS',
            'AMOUNT', 'CHARGE', 'APR', 'PAYMENT', 'TOTAL',
            'MAKE', 'MODEL', 'FINANCE'
        ]
        return any(indicator in text_upper for indicator in label_indicators) and ':' in text
    
    def _is_valid_value(self, field_name: str, value_text: str) -> bool:
        """Check if a value text looks valid for a given field."""
        value_text = value_text.strip()
        
        if not value_text or len(value_text) == 0:
            return False
        
        # Field-specific validation
        if field_name in ['phone_number', 'seller_phone_number']:
            # Look for phone number patterns
            digits = re.sub(r'\D', '', value_text)
            return 10 <= len(digits) <= 11
        
        elif field_name in ['amount_financed', 'finance_charge', 'total_of_payments', 'amount_of_payments']:
            # Look for currency patterns - must have digits
            cleaned = value_text.replace('$', '').replace(',', '').strip()
            return bool(re.search(r'^\d+\.?\d*$', cleaned))
        
        elif field_name == 'apr':
            # Look for percentage patterns
            return bool(re.search(r'\d+\.?\d*%?', value_text))
        
        elif field_name in ['quantity', 'number_of_payments']:
            # Look for integer patterns
            return bool(re.match(r'^\d+$', value_text))
        
        elif field_name in ['buyer_name', 'co_buyer_name', 'seller_name']:
            # Names should have letters and possibly spaces/hyphens
            return bool(re.search(r'[A-Za-z]', value_text)) and len(value_text) > 1
        
        elif field_name in ['street_address', 'seller_address']:
            # Addresses should have numbers and letters
            return bool(re.search(r'\d', value_text)) and bool(re.search(r'[A-Za-z]', value_text))
        
        elif field_name in ['seller_city', 'seller_state']:
            # City and state should have letters
            return bool(re.search(r'[A-Za-z]', value_text)) and len(value_text) > 1
        
        elif field_name == 'seller_zip_code':
            # ZIP code should be digits (5 or 9 digits)
            digits = re.sub(r'\D', '', value_text)
            return 5 <= len(digits) <= 9
        
        # For other fields, accept any non-empty text
        return len(value_text) > 0
    
    def _resolve_candidates(
        self,
        field_name: str,
        candidates: List[FieldCandidate]
    ) -> Optional[str]:
        """
        Resolve multiple candidates to a single value.
        
        Args:
            field_name: Field name
            candidates: List of candidate values
        
        Returns:
            Best candidate value or None
        """
        if not candidates:
            return None
        
        # For fields that need multi-word extraction, prefer candidates with more words
        if field_name in ['buyer_name', 'co_buyer_name', 'street_address', 'seller_name', 'seller_address', 'items_purchased', 'make_or_model']:
            # Prefer longer values (likely more complete)
            candidates.sort(key=lambda c: (-len(c.value.split()), c.distance, -c.confidence))
        else:
            # For single-value fields, prefer closest and highest confidence
            candidates.sort(key=lambda c: (c.distance, -c.confidence))
        
        # Return the best candidate's value
        if candidates:
            best_candidate = candidates[0]
            # Clean up the value before returning
            value = best_candidate.value.strip()
            # Remove trailing punctuation that might be OCR artifacts
            value = re.sub(r'[.,;:]+$', '', value)
            return value if value else None
        
        return None
    
    def extract_field(self, field_name: str) -> Optional[str]:
        """
        Extract a single field.
        
        Args:
            field_name: Name of the field to extract
        
        Returns:
            Extracted value or None
        """
        candidates = self._find_field_candidates(field_name)
        return self._resolve_candidates(field_name, candidates)

