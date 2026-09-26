import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_clocs_yolo_train_frame import validate_export_provenance
from scripts.export_clocs_dense_train_frame import check_training_frame


class TestOneFrameTrainingGate(unittest.TestCase):
    def test_only_frozen_train_members_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val = root / "train.txt", root / "val.txt"
            train.write_text("000000\n", encoding="utf-8")
            val.write_text("000001\n", encoding="utf-8")
            manifest = root / "split.json"
            manifest.write_text(json.dumps({
                "train": {"count": 1, "unique_count": 1,
                          "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
                "validation": {"count": 1, "unique_count": 1,
                               "sha256": hashlib.sha256(val.read_bytes()).hexdigest()},
                "overlap_count": 0, "union_count": 2,
            }), encoding="utf-8")
            self.assertTrue(check_training_frame("000000", manifest, train, val)["ok"])
            with self.assertRaisesRegex(ValueError, "not a frozen SECOND training frame"):
                check_training_frame("000001", manifest, train, val)
            train.write_text("000002\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen SECOND split mismatch"):
                check_training_frame("000000", manifest, train, val)

    def test_dense_export_requires_pinned_checkpoint_config_and_split(self):
        expected = {
            "model": {"checkpoint_sha256": "a" * 64,
                      "config": "configs/pointpillars/frozen.py"},
        }
        archive = {
            "checkpoint_sha256": np.array("a" * 64),
            "config_path": np.array("configs/pointpillars/frozen.py"),
            "config_sha256": np.array("b" * 64),
            "split_manifest_sha256": np.array("c" * 64),
        }
        self.assertEqual(validate_export_provenance(archive, expected, "c" * 64),
                         "b" * 64)
        for field, value, message in (
            ("checkpoint_sha256", "d" * 64, "checkpoint"),
            ("config_path", "configs/other.py", "config"),
            ("config_sha256", "invalid", "config SHA-256"),
            ("split_manifest_sha256", "e" * 64, "split manifest"),
        ):
            wrong = dict(archive, **{field: np.array(value)})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_export_provenance(wrong, expected, "c" * 64)


if __name__ == "__main__":
    unittest.main()
