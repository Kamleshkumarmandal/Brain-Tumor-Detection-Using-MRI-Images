"""Data loading and image preprocessing for the MRI SVM demo.

This project is an educational binary/multiclass image-classification exercise.
It is not a clinical diagnostic system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_SIZE = 128


def preprocess_image(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Convert one image to normalized grayscale features for the SVM."""
    if image_size < 16 or image_size > 512:
        raise ValueError("image_size must be between 16 and 512")
    grayscale = image.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    return pixels.reshape(1, -1)


def load_image_file(path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess an image, raising a useful error for invalid files."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            return preprocess_image(image, image_size)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not read image file: {image_path.name}") from exc


def find_dataset_directory(dataset_path: str | Path) -> Path:
    """Accept either a class-folder directory or a Kaggle-style dataset root."""
    root = Path(dataset_path).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Download the dataset and pass its folder path."
        )
    training = root / "Training"
    if training.is_dir():
        return training
    return root


def load_dataset(
    dataset_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Load image folders where each immediate subfolder is one class.

    Example:
        data/brain_tumor/Training/no_tumor/*.jpg
        data/brain_tumor/Training/pituitary_tumor/*.jpg

    If the downloaded dataset contains more class folders, they are included
    and recorded in the saved artifact. No class meaning is inferred.
    """
    directory = find_dataset_directory(dataset_path)
    class_directories = sorted(path for path in directory.iterdir() if path.is_dir())
    if len(class_directories) < 2:
        raise ValueError(
            f"Expected at least two class folders inside {directory}. "
            "Each class must be an immediate subfolder."
        )

    class_names = [path.name for path in class_directories]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped_files: list[str] = []

    for class_name, class_dir in class_to_index.items():
        folder = directory / class_name
        image_files = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(image_files) < 2:
            raise ValueError(
                f"Class folder '{class_name}' needs at least two readable images "
                "to support a stratified train/test split."
            )
        readable = 0
        for image_path in image_files:
            try:
                features.append(load_image_file(image_path, image_size)[0])
                labels.append(class_dir)
                readable += 1
            except ValueError:
                skipped_files.append(str(image_path))
        if readable < 2:
            raise ValueError(
                f"Class folder '{class_name}' has fewer than two readable images."
            )

    if len(set(labels)) < 2:
        raise ValueError("The dataset must contain at least two classes.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) != len(y):
        raise RuntimeError("Dataset feature/label count mismatch.")
    return x, y, class_names, {
        "total_images": len(x),
        "skipped_images": len(skipped_files),
        "class_count": len(class_names),
    }
