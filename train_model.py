"""Train and evaluate the MRI image SVM model.

Preferred (separate Kaggle-style Testing folder):
    python train_model.py --data-dir data/brain_tumor/Training --test-dir data/brain_tumor/Testing

Fallback (stratified split from Training):
    python train_model.py --data-dir data/brain_tumor/Training
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import ceil
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
    test_data_dir: str | Path | None = None,
) -> dict:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")

    x, y, class_names, data_summary = load_dataset(data_dir, image_size=image_size)
    counts = {name: int((y == index).sum()) for index, name in enumerate(class_names)}
    if min(counts.values()) < 2:
        raise ValueError("Every training class needs at least two images.")

    if test_data_dir is not None:
        # Prefer a distinct Testing folder when the dataset provides one.
        x_train, y_train = x, y
        x_test, y_test, test_classes, test_summary = load_dataset(
            test_data_dir, image_size=image_size
        )
        if test_classes != class_names:
            raise ValueError(
                "Training and testing class folders do not match. "
                f"Training: {class_names}; Testing: {test_classes}"
            )
        evaluation_method = "separate_test_directory"
    else:
        # Fallback: create a deterministic stratified holdout from the training folder.
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
        test_summary = {"total_images": len(x_test), "class_count": len(class_names)}
        evaluation_method = "stratified_split_from_training"

    # Scaling is fit only on training images, never on the evaluation images.
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
        "test_size": test_size if test_data_dir is None else None,
        "evaluation_method": evaluation_method,
        "training_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "class_counts": counts,
        "dataset_summary": data_summary,
        "test_summary": test_summary,
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
        "evaluation_method": evaluation_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SVM classifier on brain MRI image folders.")
    parser.add_argument("--data-dir", default="data/brain_tumor/Training", help="Training root or folder containing class directories.")
    parser.add_argument("--test-dir", default=None, help="Optional separate Testing folder containing the same class directories.")
    parser.add_argument("--output", default="models/brain_tumor_svm.joblib", help="Where to save the trained model.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--test-size", type=float, default=0.2, help="Used only when --test-dir is omitted.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    results = train(
        data_dir=args.data_dir,
        model_output=args.output,
        image_size=args.image_size,
        test_size=args.test_size,
        random_state=args.random_state,
        test_data_dir=args.test_dir,
    )
    print(f"Saved model: {results['artifact_path']}")
    print(f"Classes: {', '.join(results['class_names'])}")
    print(f"Train images: {results['training_images']} | Evaluation images: {results['test_images']}")
    print(f"Evaluation method: {results['evaluation_method']}")
    print(f"Evaluation accuracy: {results['accuracy']:.4f}")
    print("Per-class evaluation:")
    for label, metrics in results["classification_report"].items():
        if isinstance(metrics, dict):
            print(
                f"  {label}: precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}"
            )


if __name__ == "__main__":
    main()
