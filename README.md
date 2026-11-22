# Brain Tumor MRI Image Classification — SVM & Streamlit Educational Demo

An educational computer-vision project that loads MRI images from class folders, trains a scikit-learn Support Vector Machine (SVM), evaluates it on a stratified holdout set, and provides a Streamlit interface for image preview and optional model-class output.

> **Medical safety:** This repository is an academic experiment, not a medical device. It is not validated for clinical use and must not be used to diagnose, rule out, screen for, or make treatment decisions about a brain tumor. A prediction is only a dataset-class label. MRI interpretation belongs to qualified medical professionals using validated clinical workflows.

## Project status

- Image upload and preview: available in Streamlit.
- Training pipeline: available; requires you to download the dataset.
- Model artifact: **not included**. Train it locally before prediction is enabled.
- Streamlit app with no model artifact: runs in preview-only mode and does not invent predictions.
- No dataset, patient scans or generated model file are committed to this public repository.

## Dataset

The original notebook refers to the Brain Tumor Classification MRI dataset on Kaggle:

[Brain Tumor Classification MRI — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

Download the dataset yourself and place its folders in this structure:

```text
data/
└── brain_tumor/
    ├── Training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/
```

The pipeline uses the class folders found directly inside `Training`. It discovers folder names dynamically and does not assume a fixed list of classes. If your dataset uses different class names or folder names, preserve the same one-folder-per-class layout.

## Quick start (Windows PowerShell)

Use Python 3.11 or 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Train the model

From the repository root:

```powershell
python train_model.py --data-dir data/brain_tumor --output models/brain_tumor_svm.joblib
```

The training script:
- Reads the class folders inside `Training`.
- Converts images to grayscale, resizes them to 128 × 128 and normalizes pixel values to 0–1.
- Uses a stratified train/test split with a fixed random seed.
- Fits feature scaling only on the training split using a scikit-learn Pipeline.
- Trains a class-balanced RBF SVM.
- Reports holdout accuracy, per-class precision/recall/F1 and the confusion matrix.
- Saves the fitted preprocessing + classifier, class names and training metadata into a single model artifact.

The script prints the measured metrics for the dataset you train with. **No accuracy score is claimed in this README**, because the model has not been trained and evaluated from the dataset inside this repository.

### 2. Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. In the sidebar:
- **Home:** shows whether a model artifact is present.
- **MRI image analysis:** previews an uploaded image and, if the model artifact exists, displays the predicted dataset class.
- **Train a model:** shows the commands and expected folder structure.
- **About & limitations:** explains the pipeline and its limitations.

The app looks for `models/brain_tumor_svm.joblib`. Without that file it intentionally stays in preview-only mode.

## Deploy the preview app to Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io) with GitHub.
2. Choose **Create app** and select this repository.
3. Repository: `Kamleshkumarmandal/Brain-Tumor-Detection-Using-MRI-Images`.
4. Branch: `main`.
5. Main file path: `streamlit_app.py`.
6. Click **Deploy**.

The deployed repository does not contain the dataset or model artifact, so the cloud app will initially run in preview-only mode. To enable predictions in a hosted deployment, you must arrange secure access to a model artifact you trained and trust. Do not commit patient scans, sensitive health data, or untrusted joblib/pickle files to a public repository.

## Repository structure

```text
brain_tumor.py       # Image preprocessing and class-folder dataset loader
train_model.py       # Reproducible SVM training and evaluation
inference.py         # Trusted model-artifact loader and prediction helper
streamlit_app.py     # Image upload, preview and optional prediction UI
requirements.txt     # Python dependencies
tests/               # Synthetic-image pipeline and Streamlit smoke tests
models/              # Local model output (ignored by Git)
data/                # Local dataset (ignored by Git)
Brain-Tumor-Detection-Using-MRI-Images-main/
  PROJECT Brain Tumor Classification-checkpoint.ipynb
                       # Original notebook retained as a legacy reference
```

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests use generated synthetic images to verify preprocessing, class discovery, corrupt-image handling, model save/load, prediction plumbing and Streamlit rendering. These tests verify software behaviour; they do not establish medical validity or clinical performance.

GitHub Actions runs Python compilation and the test suite on pushes and pull requests.

## Important implementation notes

- The original checkpoint notebook is retained for reference. Its old cells use a hard-coded relative data path and its PCA transform is commented out; use `train_model.py` as the maintained training entrypoint.
- The training pipeline is a baseline model. Flattened grayscale pixels can discard spatial structure and may generalize poorly to images from different scanners, protocols or populations.
- Holdout metrics from one dataset are not external validation. A clinically meaningful system would require representative data, patient-level split safeguards, external validation, calibration, bias analysis, specialist review and regulatory/safety assessment.
- `joblib` model artifacts use Python object serialization. Only load an artifact you trained or otherwise trust.

## License

No license file is currently supplied. Add an appropriate license before redistributing the code or dataset.
