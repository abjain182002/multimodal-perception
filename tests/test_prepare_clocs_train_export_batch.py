#!/usr/bin/env python3
"""Split-leakage and input-provenance gates for the train export queue."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_clocs_train_export_batch import build_export_batch


class TrainExportBatchTests(unittest.TestCase):
    def test_train_only_selection_and_missing_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "training"
            for part in ("label_2", "calib", "image_2", "velodyne"):
                (dataset / part).mkdir(parents=True)
            train_ids = ["000000", "000002", "000003", "000149"]
            validation_id = "000001"
            calibration = (
                "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                "R0_rect: 1 0 0 0 1 0 0 0 1\n"
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            )
            car = "Car 0 0 0 0 0 10 10 1.5 1.8 4 10 0 0 0\n"
            for frame, cars in {"000000": 0, "000001": 7,
                                "000002": 1, "000003": 3, "000149": 7}.items():
                (dataset / "label_2" / f"{frame}.txt").write_text(car * cars)
                (dataset / "calib" / f"{frame}.txt").write_text(calibration)
                (dataset / "image_2" / f"{frame}.png").write_bytes(b"image" + frame.encode())
                (dataset / "velodyne" / f"{frame}.bin").write_bytes(b"points" + frame.encode())
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("\n".join(train_ids) + "\n")
            val.write_text(validation_id + "\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "split.json"
            manifest.write_text(json.dumps({
                "train": {"count": 4, "unique_count": 4, "sha256": digest(train)},
                "validation": {"count": 1, "unique_count": 1, "sha256": digest(val)},
                "overlap_count": 0, "union_count": 5,
            }))
            evidence = root / "model.json"
            evidence.write_text(json.dumps({"model": {
                "checkpoint_sha256": "checkpoint", "config": "config.py",
                "repository_commit": "pinned",
            }}))
            args = (dataset, train, val, manifest, evidence)
            first = build_export_batch(*args, size=4, seed=13,
                                       include=("000000", "000149"))
            again = build_export_batch(*args, size=4, seed=13,
                                       include=("000000", "000149"))
            self.assertEqual(first, again)
            self.assertEqual([f["frame_id"] for f in first["frames"]][:2],
                             ["000000", "000149"])
            self.assertEqual({f["frame_id"] for f in first["frames"]}, set(train_ids))
            self.assertEqual(first["train_pool_by_stratum"],
                             {"zero": 1, "one_two": 1,
                              "three_six": 1, "seven_plus": 1})
            self.assertEqual(first["frames"][1]["car_gt_after_range_filter"], 7)
            self.assertEqual(
                first["frames"][1]["input_sha256"]["label_2"],
                digest(dataset / "label_2" / "000149.txt"),
            )
            with self.assertRaisesRegex(ValueError, "not in the frozen training split"):
                build_export_batch(*args, size=2, include=(validation_id,))
            (dataset / "velodyne" / "000149.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                build_export_batch(*args, size=2,
                                   include=("000000", "000149"))


if __name__ == "__main__":
    unittest.main()
