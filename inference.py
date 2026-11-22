"""Safe model artifact loading and image prediction helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from brain_tumor import DEFAULT_IMAGE_SIZE, preprocess_image


def load_model_artifact(path: str | Path = "models/brain_tumor_svm.joblib") -> dict[str, Any] | None:
    """Load a locally trained artifact. Returns None when no artifact exists."""
    model_path = Path(path)
    if not model_path.is_file():
        return None
    artifact = joblib.load(model_path)
    required = {"model", "class_names", "image_size"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact is missing required metadata.")
    if not isinstance(artifact["class_names"], list) or len(artifact["class_names"]) < 2:
        raise ValueError("Model artifact has invalid class metadata.")
    return artifact


def predict_image(image, artifact: dict[str, Any]) -> str:
    """Return the model's class label, without presenting it as a diagnosis."""
    required = {"model", "class_names", "image_size"}
    if not required.issubset(artifact):
        raise ValueError("Model artifact is incomplete.")
    features = preprocess_image(image, int(artifact.get("image_size", DEFAULT_IMAGE_SIZE)))
    prediction = int(artifact["model"].predict(features)[0])
    names = artifact["class_names"]
    if prediction < 0 or prediction >= len(names):
        raise ValueError("Model returned a class outside the saved class list.")
    return str(names[prediction])
