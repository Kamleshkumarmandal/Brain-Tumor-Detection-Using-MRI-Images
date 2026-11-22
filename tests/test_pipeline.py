from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from brain_tumor import load_dataset, load_image_file, preprocess_image
from inference import load_model_artifact, predict_image
from train_model import train


def make_sample_dataset(root: Path, images_per_class: int = 6) -> Path:
    training = root / "Training"
    classes = {
        "class_alpha": 35,
        "class_beta": 210,
    }
    for class_name, base_value in classes.items():
        folder = training / class_name
        folder.mkdir(parents=True)
        for index in range(images_per_class):
            pixels = np.full((24, 24), base_value + index, dtype=np.uint8)
            # Add a small deterministic pattern so each file is not identical.
            pixels[index : index + 4, index : index + 4] = 255 - base_value
            Image.fromarray(pixels, mode="L").save(folder / f"sample_{index}.png")
    return root


class ImagePreprocessingTests(unittest.TestCase):
    def test_preprocess_image_returns_normalized_flattened_features(self):
        image = Image.new("RGB", (20, 30), color=(255, 0, 0))
        features = preprocess_image(image, image_size=16)
        self.assertEqual(features.shape, (1, 16 * 16))
        self.assertGreaterEqual(float(features.min()), 0.0)
        self.assertLessEqual(float(features.max()), 1.0)

    def test_invalid_image_size_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_image(Image.new("L", (10, 10)), image_size=8)

    def test_dataset_loader_finds_training_folder_and_class_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = make_sample_dataset(Path(temp))
            x, y, classes, summary = load_dataset(root, image_size=16)
            self.assertEqual(x.shape, (12, 256))
            self.assertEqual(len(y), 12)
            self.assertEqual(classes, ["class_alpha", "class_beta"])
            self.assertEqual(summary["class_count"], 2)

    def test_dataset_loader_skips_corrupt_image_and_counts_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = make_sample_dataset(Path(temp), images_per_class=3)
            corrupt = root / "Training" / "class_alpha" / "broken.png"
            corrupt.write_bytes(b"not an image")
            x, y, classes, summary = load_dataset(root, image_size=16)
            self.assertEqual(len(x), len(y))
            self.assertEqual(summary["skipped_images"], 1)
            self.assertEqual(len(classes), 2)

    def test_missing_dataset_directory_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset directory not found"):
                load_dataset(Path(temp) / "missing")

    def test_single_class_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "Training" / "only_class"
            folder.mkdir(parents=True)
            Image.new("L", (20, 20), color=100).save(folder / "one.png")
            Image.new("L", (20, 20), color=120).save(folder / "two.png")
            with self.assertRaisesRegex(ValueError, "at least two class folders"):
                load_dataset(Path(temp), image_size=16)

    def test_load_image_file_rejects_invalid_file(self):
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp) / "broken.jpg"
            invalid.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Could not read image file"):
                load_image_file(invalid, image_size=16)


class TrainingAndInferenceTests(unittest.TestCase):
    def test_train_save_load_and_predict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = make_sample_dataset(Path(temp) / "dataset", images_per_class=6)
            artifact_path = Path(temp) / "models" / "model.joblib"
            results = train(
                data_dir=root,
                model_output=artifact_path,
                image_size=16,
                test_size=0.25,
                random_state=42,
            )
            self.assertTrue(artifact_path.is_file())
            self.assertEqual(results["training_images"] + results["test_images"], 12)
            self.assertEqual(set(results["class_names"]), {"class_alpha", "class_beta"})
            artifact = load_model_artifact(artifact_path)
            self.assertIsNotNone(artifact)
            prediction = predict_image(Image.new("L", (24, 24), color=40), artifact)
            self.assertIn(prediction, artifact["class_names"])


    def test_train_uses_separate_testing_directory_when_provided(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training_root = make_sample_dataset(root / "dataset_train", images_per_class=5)
            testing_root = make_sample_dataset(root / "dataset_test", images_per_class=3)
            test_dir = testing_root / "Training"
            test_dir.rename(testing_root / "Testing")
            artifact_path = root / "models" / "separate-test.joblib"
            results = train(
                data_dir=training_root,
                test_data_dir=testing_root / "Testing",
                model_output=artifact_path,
                image_size=16,
            )
            self.assertEqual(results["evaluation_method"], "separate_test_directory")
            self.assertEqual(results["training_images"], 10)
            self.assertEqual(results["test_images"], 6)
            artifact = load_model_artifact(artifact_path)
            self.assertEqual(artifact["evaluation_method"], "separate_test_directory")

    def test_train_rejects_too_small_stratified_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = make_sample_dataset(Path(temp), images_per_class=2)
            with self.assertRaisesRegex(ValueError, "Not enough images for a stratified split"):
                train(root, Path(temp) / "model.joblib", image_size=16, test_size=0.2)

    def test_missing_artifact_returns_none(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(load_model_artifact(Path(temp) / "missing.joblib"))


if __name__ == "__main__":
    unittest.main()
