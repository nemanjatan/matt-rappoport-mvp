"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from src.pipeline import ExtractionPipeline

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "examples"
CREDENTIALS_PATH = PROJECT_ROOT / "matt-481014-e5ff3d867b2a.json"


@pytest.fixture
def pipeline():
    """Create extraction pipeline instance."""
    return ExtractionPipeline(
        credentials_path=str(CREDENTIALS_PATH) if CREDENTIALS_PATH.exists() else None
    )


@pytest.fixture
def pipeline_with_openai():
    """Create extraction pipeline with OpenAI (if available)."""
    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ExtractionPipeline(
            credentials_path=str(CREDENTIALS_PATH) if CREDENTIALS_PATH.exists() else None,
            openai_api_key=openai_key,
            force_openai=False
        )
    return None


@pytest.fixture
def img_1805_path():
    """Path to IMG_1805.png test image."""
    path = TEST_IMAGES_DIR / "IMG_1805.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture
def img_1807_path():
    """Path to IMG_1807.png test image."""
    path = TEST_IMAGES_DIR / "IMG_1807.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return str(path)


@pytest.fixture
def expected_img_1805():
    """Expected extraction results for IMG_1805.png."""
    return {
        "buyer_name": "Hannah Hornberger",
        "street_address": "500 Ricky Street, Sene, CA 71760",
        "phone_number": "717-257-0626",
        "co_buyer_name": "Randy Hornberger",
        "quantity": 2,
        "items_purchased": "Appliances",
        "make_or_model": "Platinum Couture / Cutler",
        "amount_financed": 6998.00,
        "finance_charge": 3607.72,
        "apr": 21.0,
        "total_of_payments": 11025.76,
        "number_of_payments": 48,
        "amount_of_payments": 229.70
    }


@pytest.fixture
def expected_img_1807():
    """Expected extraction results for IMG_1807.png."""
    return {
        "buyer_name": "David Powers",
        "street_address": "214 Cheyenne Trail, Liberty, SC 29657",
        "phone_number": "843-333-4540",
        "co_buyer_name": "Lydia Powers",
        "quantity": 1,
        "items_purchased": "Food / Goods and/or Services",
        "make_or_model": None,  # N/A
        "amount_financed": 3644.28,
        "finance_charge": 0.0,
        "apr": 0.0,
        "total_of_payments": 3644.28,
        "number_of_payments": 6,
        "amount_of_payments": 607.38
    }

