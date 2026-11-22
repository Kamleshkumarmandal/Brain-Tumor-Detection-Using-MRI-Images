# Brain Tumor MRI Image Classification — SVM & Streamlit Educational Demo

An educational computer-vision project that loads MRI images from class folders, trains a scikit-learn Support Vector Machine (SVM), evaluates it, and provides a Streamlit interface for image preview and optional model-class output.

> **Medical safety:** This repository is an academic experiment, not a medical device. It is not validated for clinical use and must not be used to diagnose, rule out, screen for, or make treatment decisions about a brain tumor. A prediction is only a dataset-class label. MRI interpretation belongs to qualified medical professionals using validated clinical workflows.

## Project status
🚀live demo
https://brain-tumor-detection-using-mri-images-hourma4g3xbqstkjbmh3yd.streamlit.app/

- Image upload and preview: available in Streamlit.
- Training pipeline: available; requires you to download the dataset.
- Model artifact: **not included**. Train it locally before prediction is enabled.
- Streamlit app without a model artifact: runs in preview-only mode and does not invent predictions.
- No dataset, patient scans or generated model file are committed to this public repository.

## Dataset

The original notebook refers to the [Brain Tumor Classification MRI dataset on Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).

Download the dataset yourself and arrange it in this structure:

```text
data/
└── brain_tumor/
    ├── Training/
    │   ├── class_one/
    │   ├── class_two/
    │   └── ... one folder per dataset class
    └── Testing/
        ├── class_one/
        ├── class_two/
        └── ... same class-folder names as Training
```

The original notebook's binary experiment defines the labels `no_tumor` and `pituitary_tumor`. The maintained training pipeline discovers the folder names dynamically, so it can also use a dataset with additional classes. Confirm the folder names in the dataset you downloaded; the model will report those names as its output labels.

## Quick start (Windows PowerShell)

Use Python 3.11 or 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Train and evaluate the model

When the dataset contains both `Training` and `Testing`:

```powershell
python train_model.py --data-dir data/brain_tumor/Training --test-dir data/brain_tumor/Testing --output models/brain_tumor_svm.joblib
```

If you only have class folders and no separate Testing folder, use a stratified split from the training directory:

```powershell
python train_model.py --data-dir data/brain_tumor/Training --output models/brain_tumor_svm.joblib
```

The training script:
- Discovers image classes from immediate subfolders.
- Converts images to grayscale, resizes them to 128 × 128 and normalizes pixels to 0–1.
- Uses a separate Testing folder when supplied; otherwise uses a deterministic stratified split.
- Fits feature scaling only on training images using a scikit-learn Pipeline.
- Trains a class-balanced RBF SVM.
- Reports evaluation accuracy, per-class precision/recall/F1 and a confusion matrix.
- Saves preprocessing, classifier, class names and evaluation metadata into one model artifact.

The script prints metrics measured on the data you supply. **No accuracy score is claimed in this README**, because no dataset or trained model is bundled here. A dataset holdout score is not clinical validation.

### 2. Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit:
- **Home:** shows whether a model artifact is present.
- **MRI image analysis:** previews an uploaded image and, if the local artifact exists, displays the model's predicted dataset class.
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

The public repository contains neither the dataset nor the model artifact, so the cloud app will initially run in preview-only mode. To enable predictions in a hosted deployment, arrange secure access to a model artifact you trained and trust. Do not commit patient scans, sensitive health data, or untrusted joblib/pickle files to a public repository.

## Repository structure

```text
brain_tumor.py       # Image preprocessing and class-folder dataset loader
train_model.py       # Reproducible SVM training and evaluation
inference.py         # Trusted model-artifact loader and prediction helper
streamlit_app.py     # Image upload, preview and optional model output
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

Tests use generated synthetic images to verify preprocessing, class discovery, corrupt-image handling, model save/load, prediction plumbing, separate test-folder evaluation and Streamlit rendering. They test software behaviour; they do not establish medical validity or clinical performance.

GitHub Actions runs Python compilation and the test suite on pushes and pull requests.

## Important implementation notes

- The original checkpoint notebook is retained for reference. Its old cells use a hard-coded relative data path, only map two class names, and have the PCA transform commented out. Use `train_model.py` as the maintained training entrypoint.
- The fallback split is image-level, not patient-level. If related images from the same patient can appear in both partitions, performance can be overestimated. The dataset does not provide verified patient-level identifiers in this repository.
- Flattened grayscale pixels can discard spatial structure and may generalize poorly across scanners, protocols or populations.
- Holdout metrics from one dataset are not external validation. A clinically meaningful system would require representative data, patient-level split safeguards, external validation, calibration, bias analysis, specialist review and regulatory/safety assessment.
- `joblib` artifacts use Python object serialization. Only load an artifact you trained or otherwise trust.

## License

No license file is currently supplied. Add an appropriate license before redistributing the code or dataset.
