# Installment Agreement Schema

Canonical schema definition for all extracted fields from installment credit agreement documents.

## Usage

```python
from src.schema import InstallmentAgreementSchema

# Create schema instance with extracted data
data = {
    "buyer_name": "John Doe",
    "phone_number": "(843) 333-4540",  # Will be normalized to "843-333-4540"
    "amount_financed": "$3,644.28",    # Will be normalized to Decimal("3644.28")
    "apr": "21%",                       # Will be normalized to Decimal("21")
    "quantity": "2",                    # Will be normalized to int(2)
    "make_or_model": "N/A"              # Will be normalized to None
}

schema = InstallmentAgreementSchema(**data)

# Access normalized values
print(schema.phone_number)      # "843-333-4540"
print(schema.amount_financed)    # Decimal("3644.28")
print(schema.apr)                # Decimal("21")
print(schema.quantity)           # 2
print(schema.make_or_model)      # None

# Convert to JSON-serializable dict
json_data = schema.to_json_dict()
```

## Field Types

- **String fields**: `buyer_name`, `co_buyer_name`, `street_address`, `phone_number`, `items_purchased`, `make_or_model`
- **Decimal fields**: `amount_financed`, `finance_charge`, `apr`, `total_of_payments`, `amount_of_payments`
- **Integer fields**: `quantity`, `number_of_payments`

## Normalization Rules

### Currency Values
- Removes `$` and commas
- Converts to `Decimal` type
- Examples: `"$3,644.28"` → `Decimal("3644.28")`, `"0.00"` → `Decimal("0.00")`

### Phone Numbers
- Normalizes to `XXX-XXX-XXXX` format
- Handles `(XXX) XXX-XXXX` and `XXXXXXXXXX` formats
- Examples: `"(843) 333-4540"` → `"843-333-4540"`, `"8433334540"` → `"843-333-4540"`

### APR Values
- Removes `%` symbol
- Converts to `Decimal` type
- Examples: `"21%"` → `Decimal("21")`, `"0.00%"` → `Decimal("0.00")`

### N/A Handling
- Fields accept `"N/A"`, `"NA"`, `""`, `"None"`, `"NULL"` and convert to `None`
- Applies to: `co_buyer_name`, `make_or_model`, and all optional fields

## All Fields

**Seller Information:**
1. `seller_name` (string, optional)
2. `seller_address` (string, optional)
3. `seller_city` (string, optional)
4. `seller_state` (string, optional) - 2-letter state code
5. `seller_zip_code` (string, optional)
6. `seller_phone_number` (string, optional) - normalized to XXX-XXX-XXXX

**Buyer Information:**
7. `buyer_name` (string, optional)
8. `co_buyer_name` (string, optional)
9. `street_address` (string, optional)
10. `phone_number` (string, optional) - normalized to XXX-XXX-XXXX

**Purchase Details:**
11. `quantity` (integer, optional)
12. `items_purchased` (string, optional)
13. `make_or_model` (string, optional) - N/A values converted to None

**Financial Data:**
14. `amount_financed` (decimal, optional)
15. `finance_charge` (decimal, optional)
16. `apr` (decimal, optional) - percentage as decimal (e.g., 21 for 21%)
17. `total_of_payments` (decimal, optional)

**Payment Details:**
18. `number_of_payments` (integer, optional)
19. `amount_of_payments` (decimal, optional)

