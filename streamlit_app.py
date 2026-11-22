
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection Using MRI Images")
st.write("MRI Image Analysis Dashboard")

st.info(
    "Upload an MRI image to preview it. "
    "The trained detection model must be connected "
    "before diagnostic predictions can be displayed."
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "MRI Image Analysis", "About Project"]
)

if page == "Home":
    st.header("Welcome")
    st.write(
        "This project demonstrates a brain MRI image "
        "upload and analysis interface."
    )

    col1, col2 = st.columns(2)
    col1.metric("Image Input", "Supported")
    col2.metric("AI Model", "Not connected")

    st.subheader("Project Features")
    st.markdown("""
    - Upload MRI brain images
    - Preview uploaded images
    - Prepare images for model analysis
    - Connect a trained deep learning model
    """)

elif page == "MRI Image Analysis":
    st.header("Upload Brain MRI Image")

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI Image")
        st.image(
            image,
            caption="MRI Image Preview",
            use_container_width=True
        )

        st.success("Image uploaded successfully!")
        st.warning(
            "No tumor classification is available yet. "
            "Connect the trained model to enable predictions."
        )
    else:
        st.info("Please upload an MRI image to continue.")

elif page == "About Project":
    st.header("About the Project")

    st.write("""
    Brain Tumor Detection Using MRI Images is an
    AI-based project intended to analyze brain MRI scans.

    The interface is prepared for integration with a
    trained machine learning or deep learning model.
    """)

    st.subheader("Technology Stack")
    st.write("""
    - Python
    - Streamlit
    - Pillow
    - Deep Learning model (to be connected)
    """)

st.divider()
st.caption(
    "Academic demonstration only. Not intended for medical diagnosis."
)
