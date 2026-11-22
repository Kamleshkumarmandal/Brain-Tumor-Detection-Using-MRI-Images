from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import load_model_artifact, predict_image

st.set_page_config(page_title="Brain MRI Classifier | Educational Demo", page_icon="🧠", layout="wide")

MODEL_PATH = Path("models/brain_tumor_svm.joblib")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@st.cache_resource(show_spinner=False)
def load_cached_model(path: str, modified_time: float):
    # The modification time changes the cache key after a local retraining run.
    del modified_time
    return load_model_artifact(path)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f6f8fb;color:#1d2940}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e8edf2}
.brand{font:800 24px Manrope;letter-spacing:-1px;color:#1d2940;margin:5px 0 2px}
.brand span{color:#6557d9}
.brand-sub{font-size:8px;letter-spacing:1.2px;color:#929daf;font-weight:800;margin-bottom:23px}
.hero{background:linear-gradient(115deg,#20294c,#454179);border-radius:16px;padding:25px 28px;color:white;margin:7px 0 17px}
.hero .eyebrow{color:#c8c2ff;font-size:9px;letter-spacing:1.5px;font-weight:800;margin-bottom:10px}
.hero h1{font:800 30px Manrope;color:#fff;letter-spacing:-1.3px;margin:0 0 8px}
.hero p{color:#e0e1ef;font-size:12px;line-height:1.7;margin:0;max-width:740px}
.section-title{font:800 19px Manrope;letter-spacing:-.4px;color:#24304a;margin:10px 0 4px}
.section-sub{font-size:10px;color:#8c98aa;margin-bottom:13px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e6ebf2;border-radius:11px;padding:13px 15px}
div[data-testid="stMetricLabel"]{font-size:10px;color:#8793a6}
div[data-testid="stMetricValue"]{font:800 22px Manrope;color:#24304a}
.notice{background:#efedff;border:1px solid #e3e0ff;border-radius:10px;padding:12px 15px;color:#5549bb;font-size:11px;line-height:1.6;margin:10px 0 15px}
.disclaimer{background:#fff8e9;border:1px solid #f3e5c4;border-radius:10px;padding:12px 15px;color:#765a25;font-size:10px;line-height:1.65;margin:12px 0}
.smallcaps{font-size:8px;letter-spacing:1.1px;color:#9ba5b4;font-weight:800}
.footer{font-size:9px;color:#9aa5b4;text-align:center;padding:20px 0 8px}
.stButton>button{border-radius:8px;font-weight:600}
.stButton>button[kind="primary"]{background:#6557d9;border-color:#6557d9}
hr{border-color:#e7ebf2}
</style>
""", unsafe_allow_html=True)

artifact = None
model_error = None
try:
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.is_file() else 0.0
    if model_mtime:
        artifact = load_cached_model(str(MODEL_PATH), model_mtime)
except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"

with st.sidebar:
    st.markdown('<div class="brand">neuro<span>scan</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">BRAIN MRI CLASSIFICATION · DEMO</div>', unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Home", "MRI image analysis", "Train a model", "About & limitations"])
    st.divider()
    if artifact:
        st.success("Local model artifact found")
        st.caption("Classes: " + ", ".join(artifact["class_names"]))
    elif model_error:
        st.error("Model file could not be loaded")
    else:
        st.warning("No trained model found")
    st.caption("Educational image-classification project. Not a medical device.")

st.markdown(
    '<div class="hero"><div class="eyebrow">IMAGE INPUT · REPRODUCIBLE MODEL · TRANSPARENT OUTPUT</div>'
    '<h1>Brain MRI Image Classification</h1>'
    '<p>Upload an MRI image for preprocessing and, when a trained model artifact is installed, '
    'view the model dataset-class output. The interface does not present predictions as diagnoses.</p></div>',
    unsafe_allow_html=True,
)

if page == "Home":
    st.markdown('<div class="section-title">Project status</div>', unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Image input","Ready","JPG, JPEG, PNG")
    b.metric("Model status","Connected" if artifact else "Not connected","Local artifact" if artifact else "Train model first")
    c.metric("Classifier","SVM","Scikit-learn")
    if artifact:
        st.success(f"Loaded model with {len(artifact['class_names'])} dataset classes.")
        st.caption(f"Training time: {artifact.get('trained_at', 'Not recorded')} · Holdout images: {artifact.get('test_images', 'Unknown')}")
    elif model_error:
        st.error("The model artifact was found but could not be loaded: " + model_error)
    else:
        st.markdown(
            '<div class="notice"><b>Preview mode</b><br/>No trained model file is included in this repository. '
            'You can preview an image, but the app will not invent a classification. '
            'Train a model locally to enable model output.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-title">Workflow</div>',unsafe_allow_html=True)
    st.write("1. Download the MRI dataset and put each class in its own folder.")
    st.write("2. Run the included training script on your computer.")
    st.write("3. Save the artifact at models/brain_tumor_svm.joblib.")
    st.write("4. Run the app and view the model's dataset-class output.")

elif page == "MRI image analysis":
    st.markdown('<div class="section-title">Upload an MRI image</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Preprocessing uses grayscale, resize and normalization, matching the training pipeline.</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Choose an image",type=["jpg","jpeg","png"],help="Maximum upload size: 10 MB.")
    if uploaded is not None:
        raw=uploaded.getvalue()
        if len(raw)>MAX_UPLOAD_BYTES:
            st.error("This image is larger than 10 MB. Please choose a smaller file.")
        else:
            try:
                with Image.open(uploaded) as opened:
                    opened.verify()
                uploaded.seek(0)
                with Image.open(uploaded) as opened:
                    image=opened.convert("RGB")
                left,right=st.columns([1.1,1])
                with left:
                    st.image(image,caption="Uploaded MRI image",width="stretch")
                with right:
                    st.markdown("**Image details**")
                    st.write(f"Dimensions: {image.width} × {image.height} pixels")
                    st.write(f"File type: {uploaded.type or 'Unknown'}")
                    st.write(f"File size: {len(raw)/1024:.1f} KB")
                    if artifact:
                        st.markdown('<div class="section-title">Model output</div>',unsafe_allow_html=True)
                        try:
                            label=predict_image(image,artifact)
                            st.success(f"Predicted dataset class: {label}")
                            st.caption("This is only the classifier's dataset label, not a finding about a patient's health.")
                        except Exception as exc:
                            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
                    elif model_error:
                        st.error("Model could not be loaded: " + model_error)
                    else:
                        st.info("Image preview is ready. Train a model and install its artifact to enable predictions.")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This is not a readable JPG, JPEG or PNG image.")
    else:
        st.info("Choose an image to preview it.")

elif page == "Train a model":
    st.markdown('<div class="section-title">Train the SVM classifier</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Training runs locally after you download the data. Dataset images and the trained model are not included in this repository.</div>',unsafe_allow_html=True)
    st.write("Expected folder structure:")
    st.code("data/brain_tumor/\n└── Training/\n    ├── class_one/\n    │   ├── image_001.jpg\n    │   └── image_002.jpg\n    └── class_two/\n        ├── image_001.jpg\n        └── image_002.jpg",language="text")
    st.write("Install and train:")
    st.code("pip install -r requirements.txt\npython train_model.py --data-dir data/brain_tumor --output models/brain_tumor_svm.joblib",language="bash")
    st.write("Launch the app:")
    st.code("streamlit run streamlit_app.py",language="bash")
    st.info("The script uses a stratified holdout split and reports accuracy, per-class precision/recall/F1 and a confusion matrix.")
    st.warning("Do not commit private patient scans or sensitive medical data to a public repository. Only load model artifacts you trained or otherwise trust.")

elif page == "About & limitations":
    st.markdown('<div class="section-title">About this project</div>',unsafe_allow_html=True)
    st.write("This is an educational image-classification experiment using grayscale MRI images, feature scaling and a support-vector classifier (SVM). Class names come from the dataset folders and are stored with the trained model.")
    st.markdown("**Included improvements**")
    st.write("- Dataset folder discovery instead of a hard-coded computer-specific path.")
    st.write("- Invalid image handling and consistent preprocessing.")
    st.write("- Stratified train/test split and preprocessing fitted only on training data.")
    st.write("- Per-class evaluation metrics and confusion matrix.")
    st.write("- Clear model-unavailable state instead of fake predictions.")
    st.markdown('<div class="disclaimer"><b>Medical limitation</b><br/>This academic prototype is not validated for clinical use. It cannot diagnose or rule out a brain tumor and must not be used for treatment decisions. MRI interpretation must be performed by qualified medical professionals using validated clinical workflows.</div>',unsafe_allow_html=True)

st.divider()
st.markdown('<div class="smallcaps">BRAIN MRI CLASSIFICATION · ACADEMIC DEMONSTRATION</div>',unsafe_allow_html=True)
st.markdown('<div class="footer">No clinical claims · No patient data stored by this demo · Model output is experimental and dataset-dependent.</div>',unsafe_allow_html=True)
