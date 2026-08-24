import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.dataset_builder.preflight import PreflightError, require_preflight
from portrait_core.models import ModelManager


class ModelManagerTestCase(unittest.TestCase):
    def test_explicit_path_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "explicit.task"
            model.write_bytes(b"model")
            with patch.dict(os.environ, {"ORION_FACE_LANDMARKER_MODEL": "ignored.task"}):
                result = ModelManager().resolve("mediapipe_face_landmarker", explicit_path=model)
        self.assertEqual(result.source, "explicit")
        self.assertEqual(result.path, model.resolve())

    def test_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "environment.task"
            model.write_bytes(b"model")
            with patch.dict(os.environ, {"ORION_FACE_LANDMARKER_MODEL": str(model)}):
                result = ModelManager().resolve("mediapipe_face_landmarker")
        self.assertEqual(result.source, "environment")

    def test_missing_and_zero_size_models_are_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ModelManager()
            missing = manager.validate(manager.resolve("mediapipe_face_landmarker", explicit_path=root / "missing.task"), initialize_backend=False)
            empty = root / "empty.task"
            empty.touch()
            zero = manager.validate(manager.resolve("mediapipe_face_landmarker", explicit_path=empty), initialize_backend=False)
        self.assertFalse(missing.valid)
        self.assertFalse(zero.valid)

    def test_preflight_blocks_before_dataset_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images"
            source.mkdir()
            output = root / "dataset"
            with self.assertRaises(PreflightError) as caught:
                require_preflight(str(source), str(output), model_path=str(root / "missing.task"), initialize_backend=False)
            self.assertFalse(output.exists())
            self.assertEqual(caught.exception.report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
