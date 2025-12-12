"""Streamlit UI for testing the extraction pipeline."""

import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from src.pipeline import ExtractionPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Installment Agreement Extractor",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Installment Agreement Data Extractor")
st.markdown("""
Extract structured data from installment credit agreement images using:
- **Google Cloud Vision API** for OCR
- **Rule-based extraction** with keyword proximity
- **OpenAI** for enhanced accuracy and validation
""")

# Configuration (from environment)
# For Railway: Use GOOGLE_APPLICATION_CREDENTIALS env var or credentials file
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "matt-481014-e5ff3d867b2a.json"
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please add it to continue.")
    st.stop()

st.divider()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # Extract button at the top
    test_image_path = st.session_state.get('test_image_path')
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG or JPG image of an installment credit agreement"
    )
    
    # Or use test image from session state
    if test_image_path and Path(test_image_path).exists():
        st.info(f"Using test image: {Path(test_image_path).name}")
        if st.button("Clear test image"):
            del st.session_state.test_image_path
            st.rerun()
    
    # Extract button
    extract_button = st.button(
        "🚀 Run Extraction",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None and not (test_image_path and Path(test_image_path).exists()))
    )
    
    # Display uploaded image
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    elif test_image_path and Path(test_image_path).exists():
        st.image(test_image_path, caption=f"Test Image: {Path(test_image_path).name}", use_container_width=True)

with col2:
    st.header("📊 Results")
    
    # Run extraction
    if extract_button:
        try:
            # Initialize pipeline (always use OpenAI, force enabled)
            # For Railway: GOOGLE_APPLICATION_CREDENTIALS may be JSON content or file path
            # VisionOCRClient will handle both cases automatically
            creds_path = None
            if credentials_path and Path(credentials_path).exists():
                # Use local file if it exists
                creds_path = credentials_path
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                # Railway sets this - may be JSON content or file path
                # VisionOCRClient will handle it
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            # If creds_path is None, VisionOCRClient will check GOOGLE_APPLICATION_CREDENTIALS env var
            
            pipeline = ExtractionPipeline(
                credentials_path=creds_path,
                openai_api_key=openai_key,
                force_openai=True  # Always use OpenAI
            )
            with st.spinner("Extracting data... This may take a few seconds."):
                try:
                    # Determine image source
                    if uploaded_file is not None:
                        # Extract from uploaded file
                        image_bytes = uploaded_file.read()
                        result = pipeline.extract_from_bytes(
                            image_bytes,
                            image_format=uploaded_file.type.split('/')[-1].upper()
                        )
                    elif test_image_path and Path(test_image_path).exists():
                        # Extract from test image
                        result = pipeline.extract(test_image_path)
                    else:
                        st.error("No image selected")
                        result = None
                    
                    if result:
                        st.session_state.extraction_result = result
                        st.success("✅ Extraction complete!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Extraction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Pipeline initialization error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    if 'extraction_result' in st.session_state:
        result = st.session_state.extraction_result
        
        # Metadata
        with st.expander("📈 Metadata", expanded=False):
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                st.metric("OpenAI Used", "Yes" if result.used_openai else "No")
                if result.validation_result and result.validation_result.used_ai:
                    st.metric("AI Corrections", len(result.validation_result.corrections_applied))
            with col_meta2:
                if result.confidence_scores:
                    word_level = result.confidence_scores.get('word_level', {})
                    if word_level:
                        st.metric("OCR Confidence", f"{word_level.get('mean', 0):.1%}")
                st.metric("OCR Words", len(result.ocr_result.word_annotations))
                
            # Show validation issues if any
            if result.validation_result and result.validation_result.issues_found:
                st.markdown("### Validation Issues Found")
                for issue in result.validation_result.issues_found:
                    st.warning(f"**{issue.field}**: {issue.description} ({issue.severity})")
            
            # Show corrections if any
            if result.validation_result and result.validation_result.corrections_applied:
                st.markdown("### Corrections Applied")
                for correction in result.validation_result.corrections_applied:
                    st.success(f"✓ {correction}")
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["📋 Fields", "📄 JSON", "🔍 Raw OCR"])
        
        with tab1:
            st.subheader("Extracted Fields")
            
            # Organize fields by category
            schema_dict = result.schema.to_json_dict()
            
            # Helper function to display field
            def display_field(label: str, value: any, format_func=None):
                """Display a field with label and value."""
                if value is None or value == "":
                    display_value = "*Not found*"
                    color = "gray"
                else:
                    if format_func:
                        display_value = format_func(value)
                    else:
                        display_value = str(value)
                    color = "black"
                
                st.markdown(f"**{label}:** <span style='color: {color}'>{display_value}</span>", unsafe_allow_html=True)
            
            # Seller Information
            st.markdown("### 🏢 Seller Information")
            seller_cols = st.columns(3)
            with seller_cols[0]:
                display_field("Seller Name", schema_dict.get('seller_name'))
                display_field("Seller Address", schema_dict.get('seller_address'))
            with seller_cols[1]:
                display_field("City", schema_dict.get('seller_city'))
                display_field("State", schema_dict.get('seller_state'))
            with seller_cols[2]:
                display_field("ZIP Code", schema_dict.get('seller_zip_code'))
                display_field("Phone Number", schema_dict.get('seller_phone_number'))
            
            st.divider()
            
            # Buyer Information
            st.markdown("### 👤 Buyer Information")
            buyer_cols = st.columns(2)
            with buyer_cols[0]:
                display_field("Buyer Name", schema_dict.get('buyer_name'))
                display_field("Buyer Address", schema_dict.get('buyer_address') or schema_dict.get('street_address'))
                display_field("Buyer Phone", schema_dict.get('buyer_phone_number') or schema_dict.get('phone_number'))
            with buyer_cols[1]:
                display_field("Co-Buyer Name", schema_dict.get('co_buyer_name'))
                display_field("Co-Buyer Address", schema_dict.get('co_buyer_address'))
                display_field("Co-Buyer Phone", schema_dict.get('co_buyer_phone_number'))
            
            st.divider()
            
            # Purchase Details
            st.markdown("### 🛒 Purchase Details")
            purchase_cols = st.columns(3)
            with purchase_cols[0]:
                display_field("Quantity", schema_dict.get('quantity'))
            with purchase_cols[1]:
                display_field("Items Purchased", schema_dict.get('items_purchased'))
            with purchase_cols[2]:
                display_field("Make or Model", schema_dict.get('make_or_model'))
            
            st.divider()
            
            # Financial Data
            st.markdown("### 💰 Financial Data")
            financial_cols = st.columns(4)
            with financial_cols[0]:
                display_field("Amount Financed", schema_dict.get('amount_financed'), 
                            lambda v: f"${float(v):,.2f}" if v else None)
            with financial_cols[1]:
                display_field("Finance Charge", schema_dict.get('finance_charge'),
                            lambda v: f"${float(v):,.2f}" if v else None)
            with financial_cols[2]:
                display_field("APR", schema_dict.get('apr'),
                            lambda v: f"{float(v):.2f}%" if v else None)
            with financial_cols[3]:
                display_field("Total of Payments", schema_dict.get('total_of_payments'),
                            lambda v: f"${float(v):,.2f}" if v else None)
            
            st.divider()
            
            # Payment Details
            st.markdown("### 💳 Payment Details")
            payment_cols = st.columns(3)
            with payment_cols[0]:
                display_field("Number of Payments", schema_dict.get('number_of_payments'))
            with payment_cols[1]:
                display_field("Amount of Payments", schema_dict.get('amount_of_payments'),
                            lambda v: f"${float(v):,.2f}" if v else None)
        
        with tab2:
            st.subheader("JSON Output")
            json_output = result.to_dict()
            st.json(json_output)
            
            # Download button
            json_str = json.dumps(json_output, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="extraction_result.json",
                mime="application/json"
            )
        
        with tab3:
            st.subheader("Raw OCR Text")
            st.text_area(
                "OCR Text",
                value=result.ocr_result.full_text,
                height=400,
                disabled=False
            )
            
            # Confidence details
            if result.confidence_scores:
                st.markdown("### Confidence Scores")
                st.json(result.confidence_scores)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Installment Agreement Data Extractor | Powered by Google Cloud Vision & OpenAI</small><br>
    <small>OpenAI is always enabled for enhanced accuracy and validation</small>
</div>
""", unsafe_allow_html=True)

