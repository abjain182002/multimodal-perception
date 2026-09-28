import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_clocs_yolo_train_frame import (
    CAMERA_WEIGHT_SHA256, require_new_outputs, validate_camera_weights,
)


class TestCameraProvenance(unittest.TestCase):
    def test_camera_identity_is_enforced_even_if_filename_is_misleading(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "yolo26x.pt"
            weights.write_bytes(b"v8 test artifact")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            with patch.dict(CAMERA_WEIGHT_SHA256,
                            {"yolov8x": digest, "yolo26x": "0" * 64}):
                self.assertEqual(validate_camera_weights("yolov8x", weights), digest)
                with self.assertRaisesRegex(ValueError, "pinned yolo26x"):
                    validate_camera_weights("yolo26x", weights)
                weights.write_bytes(b"changed artifact")
                with self.assertRaisesRegex(ValueError, "pinned yolov8x"):
                    validate_camera_weights("yolov8x", weights)

    def test_existing_pilot_artifacts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "old_rows.npz"
            existing.write_bytes(b"preserve old pilot")
            with self.assertRaises(FileExistsError):
                require_new_outputs(root / "new_targets.npz", existing)
            self.assertEqual(existing.read_bytes(), b"preserve old pilot")
            self.assertFalse((root / "new_targets.npz").exists())
            require_new_outputs(root / "new_targets.npz", root / "new_rows.npz")


if __name__ == "__main__":
    unittest.main()
