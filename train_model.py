"""Train and evaluate the MRI image SVM model.

Example:
    python train_model.py --data-dir data/brain_tumor
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from brain_tumor import DEFAULT_IMAGE_SIZE, load_dataset


def train(
    data_dir: str | Path,
    model_output: str | Path = "models/brain_tumor_svm.joblib",
    image_size: int = DEFAULT_IMAGE_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")

    x, y, class_names, data_summary = load_dataset(data_dir, image_size=image_size)
    counts = {name: int((y == index).sum()) for index, name in enumerate(class_names)}
    if min(counts.values()) < 2:
        raise ValueError("Every class needs at least two images.")

    # Stratification requires at least one sample from every class in both splits.
    from math import ceil
    test_count = ceil(len(y) * test_size)
    train_count = len(y) - test_count
    if test_count < len(class_names) or train_count < len(class_names):
        raise ValueError(
            "Not enough images for a stratified split with this test_size. "
            "Add more images per class or increase test_size."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Scaling happens inside the pipeline and is fit only on the training split.
    # This avoids leaking information from validation/test images into preprocessing.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(class_names))))

    artifact = {
        "model": model,
        "class_names": class_names,
        "image_size": image_size,
        "model_type": "scikit-learn SVC with StandardScaler",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "test_size": test_size,
        "training_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "class_counts": counts,
        "dataset_summary": data_summary,
        "test_accuracy": accuracy,
    }
    output_path = Path(model_output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    return {
        "artifact_path": str(output_path),
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "training_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "class_counts": counts,
        "dataset_summary": data_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SVM classifier on brain MRI image folders.")
    parser.add_argument("--data-dir", default="data/brain_tumor", help="Dataset root or Training class-folder path.")
    parser.add_argument("--output", default="models/brain_tumor_svm.joblib", help="Where to save the trained model.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    results = train(
        data_dir=args.data_dir,
        model_output=args.output,
        image_size=args.image_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Saved model: {results['artifact_path']}")
    print(f"Classes: {', '.join(results['class_names'])}")
    print(f"Train images: {results['training_images']} | Test images: {results['test_images']}")
    print(f"Holdout accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    for label, metrics in results["classification_report"].items():
        if isinstance(metrics, dict):
            print(
                f"  {label}: precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}"
            )


if __name__ == "__main__":
    main()
