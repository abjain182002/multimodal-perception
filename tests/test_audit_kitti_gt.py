#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_kitti_gt import audit_dataset


class TestKITTIGTAudit(unittest.TestCase):
    def test_split_separation_and_empty_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "training"
            (dataset / "calib").mkdir(parents=True)
            (dataset / "label_2").mkdir()
            calibration = (
                "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                "R0_rect: 1 0 0 0 1 0 0 0 1\n"
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            )
            for frame in ("000001", "000002"):
                (dataset / "calib" / f"{frame}.txt").write_text(calibration)
            (dataset / "label_2" / "000001.txt").write_text(
                "Car 0 0 0 0 0 10 10 1.5 1.8 4 10 0 0 0\n"
            )
            (dataset / "label_2" / "000002.txt").write_text("")

            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("000001\n")
            val.write_text("000002\n")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = root / "split.json"
            manifest.write_text(json.dumps({
                "train": {
                    "count": 1, "unique_count": 1,
                    "sha256": digest(train),
                },
                "validation": {
                    "count": 1, "unique_count": 1,
                    "sha256": digest(val),
                },
                "overlap_count": 0, "union_count": 2,
            }))

            result = audit_dataset(dataset, manifest, train, val)
            self.assertEqual(result["train"]["car_after_range_filter"], 1)
            self.assertEqual(result["validation"]["car_after_range_filter"], 0)
            self.assertEqual(
                result["validation"]["frames_without_car_after_filter"], 1
            )
            self.assertEqual(result["total"]["frames"], 2)

            val.write_text("000001\n")
            with self.assertRaisesRegex(ValueError, "Split manifest mismatch"):
                audit_dataset(dataset, manifest, train, val)


if __name__ == "__main__":
    unittest.main()
