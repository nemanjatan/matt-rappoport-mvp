"""Canonical schema for installment credit agreement extracted fields."""

from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class InstallmentAgreementSchema(BaseModel):
    """
    Canonical schema for all extracted fields from installment credit agreements.
    
    All fields support null/None values when data is explicitly missing.
    Currency values are normalized to Decimal type with '.' separator.
    Phone numbers are normalized to XXX-XXX-XXXX format.
    """
    
    # Seller Information
    seller_name: Optional[str] = Field(
        default=None,
        description="Seller's business name"
    )
    
    seller_address: Optional[str] = Field(
        default=None,
        description="Seller's street address"
    )
    
    seller_city: Optional[str] = Field(
        default=None,
        description="Seller's city"
    )
    
    seller_state: Optional[str] = Field(
        default=None,
        description="Seller's state (2-letter code)"
    )
    
    seller_zip_code: Optional[str] = Field(
        default=None,
        description="Seller's ZIP code"
    )
    
    seller_phone_number: Optional[str] = Field(
        default=None,
        description="Seller's phone number in XXX-XXX-XXXX format"
    )
    
    # Buyer Information
    buyer_name: Optional[str] = Field(
        default=None,
        description="Buyer's full name"
    )
    
    buyer_address: Optional[str] = Field(
        default=None,
        description="Buyer's street address"
    )
    
    buyer_phone_number: Optional[str] = Field(
        default=None,
        description="Buyer's phone number in XXX-XXX-XXXX format"
    )
    
    co_buyer_name: Optional[str] = Field(
        default=None,
        description="Co-buyer's full name (optional)"
    )
    
    co_buyer_address: Optional[str] = Field(
        default=None,
        description="Co-buyer's street address (optional)"
    )
    
    co_buyer_phone_number: Optional[str] = Field(
        default=None,
        description="Co-buyer's phone number in XXX-XXX-XXXX format (optional)"
    )
    
    # Legacy field names for backward compatibility
    street_address: Optional[str] = Field(
        default=None,
        description="Street address (maps to buyer_address for backward compatibility)"
    )
    
    phone_number: Optional[str] = Field(
        default=None,
        description="Phone number (maps to buyer_phone_number for backward compatibility)"
    )
    
    # Purchase Details
    quantity: Optional[int] = Field(
        default=None,
        description="Quantity of items purchased",
        ge=0
    )
    
    items_purchased: Optional[str] = Field(
        default=None,
        description="Description of items purchased"
    )
    
    make_or_model: Optional[str] = Field(
        default=None,
        description="Make or model of items (may be 'N/A')"
    )
    
    # Financial Data (Truth in Lending)
    amount_financed: Optional[Decimal] = Field(
        default=None,
        description="Amount financed in decimal format (e.g., 3644.28)"
    )
    
    finance_charge: Optional[Decimal] = Field(
        default=None,
        description="Finance charge in decimal format (e.g., 0.00)"
    )
    
    apr: Optional[Decimal] = Field(
        default=None,
        description="Annual Percentage Rate as decimal (e.g., 21.00 for 21%)",
        ge=0,
        le=100
    )
    
    total_of_payments: Optional[Decimal] = Field(
        default=None,
        description="Total of payments in decimal format"
    )
    
    # Payment Details
    number_of_payments: Optional[int] = Field(
        default=None,
        description="Number of installment payments",
        ge=0
    )
    
    amount_of_payments: Optional[Decimal] = Field(
        default=None,
        description="Amount per payment in decimal format"
    )
    
    @field_validator('phone_number', 'buyer_phone_number', 'co_buyer_phone_number', 'seller_phone_number')
    @classmethod
    def normalize_phone_number(cls, v: Optional[str]) -> Optional[str]:
        """Normalize phone number to XXX-XXX-XXXX format."""
        if v is None:
            return None
        
        # Handle N/A or empty strings
        v = v.strip().upper()
        if v in ('N/A', 'NA', '', 'NONE', 'NULL'):
            return None
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', v)
        
        # Validate length (should be 10 digits for US phone numbers)
        if len(digits) == 10:
            return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
        elif len(digits) == 11 and digits[0] == '1':
            # Handle 11-digit numbers starting with 1
            return f"{digits[1:4]}-{digits[4:7]}-{digits[7:11]}"
        else:
            # Return as-is if can't normalize (might be international or malformed)
            return v
    
    @field_validator('make_or_model')
    @classmethod
    def normalize_make_or_model(cls, v: Optional[str]) -> Optional[str]:
        """Normalize make_or_model field, handling N/A values."""
        if v is None:
            return None
        
        v = v.strip()
        if v.upper() in ('N/A', 'NA', '', 'NONE', 'NULL'):
            return None
        
        return v
    
    @field_validator('amount_financed', 'finance_charge', 'total_of_payments', 'amount_of_payments', mode='before')
    @classmethod
    def normalize_currency(cls, v) -> Optional[Decimal]:
        """Normalize currency values to Decimal."""
        if v is None:
            return None
        
        # Handle string inputs
        if isinstance(v, str):
            v = v.strip()
            # Handle N/A or empty strings
            if v.upper() in ('N/A', 'NA', '', 'NONE', 'NULL'):
                return None
            
            # Remove currency symbols and commas
            v = v.replace('$', '').replace(',', '').strip()
            
            # Handle empty after cleaning
            if not v:
                return None
        
        # Convert to Decimal
        try:
            return Decimal(str(v))
        except (ValueError, TypeError):
            return None
    
    @field_validator('apr', mode='before')
    @classmethod
    def normalize_apr(cls, v) -> Optional[Decimal]:
        """Normalize APR percentage to Decimal (e.g., '21%' -> 21.00)."""
        if v is None:
            return None
        
        # Handle string inputs
        if isinstance(v, str):
            v = v.strip()
            # Handle N/A or empty strings
            if v.upper() in ('N/A', 'NA', '', 'NONE', 'NULL'):
                return None
            
            # Remove percentage symbol and any whitespace
            v = v.replace('%', '').strip()
            
            # Handle empty after cleaning
            if not v:
                return None
        
        # Convert to Decimal
        try:
            return Decimal(str(v))
        except (ValueError, TypeError):
            return None
    
    @field_validator('quantity', 'number_of_payments', mode='before')
    @classmethod
    def normalize_integer(cls, v) -> Optional[int]:
        """Normalize integer fields."""
        if v is None:
            return None
        
        # Handle string inputs
        if isinstance(v, str):
            v = v.strip()
            # Handle N/A or empty strings
            if v.upper() in ('N/A', 'NA', '', 'NONE', 'NULL'):
                return None
        
        # Convert to int
        try:
            return int(float(str(v)))  # Handle "3.0" -> 3
        except (ValueError, TypeError):
            return None
    
    @model_validator(mode='after')
    def validate_schema(self):
        """Additional validation logic and backward compatibility mapping."""
        # Map legacy fields to new fields for backward compatibility
        if self.street_address and not self.buyer_address:
            self.buyer_address = self.street_address
        if self.phone_number and not self.buyer_phone_number:
            self.buyer_phone_number = self.phone_number
        
        return self
    
    def to_dict(self) -> dict:
        """Convert schema to dictionary with proper serialization."""
        result = {}
        for field_name, field_value in self.model_dump().items():
            if field_value is None:
                result[field_name] = None
            elif isinstance(field_value, Decimal):
                result[field_name] = str(field_value)
            else:
                result[field_name] = field_value
        return result
    
    def to_json_dict(self) -> dict:
        """Convert schema to JSON-serializable dictionary."""
        result = {}
        for field_name, field_value in self.model_dump().items():
            if field_value is None:
                result[field_name] = None
            elif isinstance(field_value, Decimal):
                # Convert Decimal to string for JSON serialization
                result[field_name] = float(field_value)
            else:
                result[field_name] = field_value
        return result
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


# Field type definitions for reference
class FieldTypes:
    """Type definitions for schema fields."""
    
    STRING = str
    DECIMAL = Decimal
    INTEGER = int
    
    # Field type mapping
    FIELD_TYPES = {
        'seller_name': STRING,
        'seller_address': STRING,
        'seller_city': STRING,
        'seller_state': STRING,
        'seller_zip_code': STRING,
        'seller_phone_number': STRING,
        'buyer_name': STRING,
        'buyer_address': STRING,
        'buyer_phone_number': STRING,
        'co_buyer_name': STRING,
        'co_buyer_address': STRING,
        'co_buyer_phone_number': STRING,
        'street_address': STRING,  # Legacy
        'phone_number': STRING,  # Legacy
        'quantity': INTEGER,
        'items_purchased': STRING,
        'make_or_model': STRING,
        'amount_financed': DECIMAL,
        'finance_charge': DECIMAL,
        'apr': DECIMAL,
        'total_of_payments': DECIMAL,
        'number_of_payments': INTEGER,
        'amount_of_payments': DECIMAL,
    }
    
    @classmethod
    def get_field_type(cls, field_name: str) -> type:
        """Get the type for a given field name."""
        return cls.FIELD_TYPES.get(field_name, str)
    
    @classmethod
    def get_all_fields(cls) -> list[str]:
        """Get all field names in the schema."""
        return list(cls.FIELD_TYPES.keys())

