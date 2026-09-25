import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fusion.clocs import build_clocs_feature_rows
from fusion.clocs_dataset import load_training_example_npz
from fusion.kitti_labels import parse_kitti_label_line
from scripts.build_clocs_training_example import build_real_example, main, read_inputs


class TestRealExampleConstruction(unittest.TestCase):
    def setUp(self):
        self.gt = parse_kitti_label_line(
            "Car 0 0 0 0 0 10 10 1.5 2 4 0 1.5 20 0")
        boxes = np.stack([self.gt["camera_box_lhw"]] * 3)
        boxes[1, 0] += 1
        boxes[2, 0] += 20
        self.candidates = {
            "boxes_camera": boxes,
            "scores_3d": np.array([0.8, 0.6, 0.2]),
            "distance_82": np.array([0.2, 0.21, 0.4]),
            "candidate_index": np.arange(3),
            "projected_boxes": np.array([[0, 0, 2, 2],
                                          [1, 0, 3, 2],
                                          [100, 0, 102, 2]], dtype=np.float32),
        }
        boxes_2d = np.array([[0, 0, 2, 2], [2, 0, 3, 2]], dtype=np.float32)
        scores_2d = np.array([0.7, 0.5], dtype=np.float32)
        rows = build_clocs_feature_rows(
            self.candidates["projected_boxes"], self.candidates["scores_3d"],
            boxes_2d, scores_2d, self.candidates["distance_82"])
        self.features = {
            "features": rows["features"],
            "candidate_index": rows["candidate_index"],
            "detector_2d_index": rows["detector_2d_index"],
            "row_type": rows["row_type"],
            "boxes_2d": boxes_2d,
            "scores_2d": scores_2d,
        }

    def test_real_geometry_targets_and_row_coverage(self):
        example, gt = build_real_example("000001", "train", self.candidates,
                                         self.features, [self.gt], np.eye(4))
        self.assertEqual(gt["num_car_after_range_filter"], 1)
        np.testing.assert_allclose(example["best_iou"], [1, 0.6, 0], atol=1e-6)
        np.testing.assert_array_equal(example["labels"], [1, -1, 0])
        self.assertEqual(example["num_rows"], 4)

    def test_empty_gt_is_valid_negative_frame(self):
        example, _ = build_real_example("000001", "train", self.candidates,
                                        self.features, [], np.eye(4))
        np.testing.assert_array_equal(example["labels"], [0, 0, 0])

    def test_reject_misaligned_candidate_order(self):
        self.candidates["candidate_index"] = np.array([1, 0, 2])
        with self.assertRaisesRegex(ValueError, "original dense box order"):
            build_real_example("000001", "train", self.candidates,
                               self.features, [self.gt], np.eye(4))

    def test_reject_misaligned_feature_score_or_distance(self):
        for column in [1, 3]:
            original = self.features["features"][0, column]
            self.features["features"][0, column] = 0.123
            with self.assertRaisesRegex(ValueError, "do not match"):
                build_real_example("000001", "train", self.candidates,
                                   self.features, [self.gt], np.eye(4))
            self.features["features"][0, column] = original

    def test_reject_misaligned_2d_detection_and_projection(self):
        for source, field, index in [
            (self.features, "boxes_2d", (0, 0)),
            (self.features, "scores_2d", (0,)),
            (self.candidates, "projected_boxes", (0, 0)),
        ]:
            original = source[field][index]
            source[field][index] += 0.25
            with self.assertRaisesRegex(ValueError, "regenerated 2D/3D associations"):
                build_real_example("000001", "train", self.candidates,
                                   self.features, [self.gt], np.eye(4))
            source[field][index] = original

    def test_reject_misaligned_2d_row_indices(self):
        self.features["detector_2d_index"][0] = 1
        with self.assertRaisesRegex(ValueError, "regenerated 2D/3D associations"):
            build_real_example("000001", "train", self.candidates,
                               self.features, [self.gt], np.eye(4))

    def test_reject_invalid_2d_detection_geometry(self):
        self.features["boxes_2d"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            build_real_example("000001", "train", self.candidates,
                               self.features, [self.gt], np.eye(4))

    def test_input_archive_checks_frame_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, frame_id="000001", scores_3d=np.array([0.3]))
            self.assertEqual(read_inputs(path, ("scores_3d",), "000001")
                             ["scores_3d"].tolist(), [0.3])
            with self.assertRaisesRegex(ValueError, "frame_id"):
                read_inputs(path, ("scores_3d",), "000002")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                read_inputs(path, ("boxes_camera",), "000001")

    def test_cli_checks_split_and_writes_verified_real_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "kitti"
            (dataset / "label_2").mkdir(parents=True)
            (dataset / "calib").mkdir()
            (dataset / "label_2" / "000001.txt").write_text(
                "Car 0 0 0 0 0 10 10 1.5 2 4 0 1.5 20 0\n")
            (dataset / "calib" / "000001.txt").write_text(
                "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                "R0_rect: 1 0 0 0 1 0 0 0 1\n"
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("000001\n")
            val.write_text("000002\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "train": {"count": 1, "unique_count": 1,
                          "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
                "validation": {"count": 1, "unique_count": 1,
                               "sha256": hashlib.sha256(val.read_bytes()).hexdigest()},
                "overlap_count": 0, "union_count": 2,
            }))
            candidate_path = root / "candidates.npz"
            feature_path = root / "features.npz"
            np.savez(candidate_path, frame_id="000001", **self.candidates)
            np.savez(feature_path, frame_id="000001", **self.features)
            output = root / "example.npz"
            args = ["build_clocs_training_example", "--dataset", str(dataset),
                    "--frame-id", "000001", "--candidates", str(candidate_path),
                    "--features", str(feature_path), "--train", str(train),
                    "--val", str(val), "--manifest", str(manifest),
                    "--output", str(output)]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            example = load_training_example_npz(output)
            self.assertEqual(example["split"], "train")
            self.assertEqual(example["num_positive"], 1)
            self.assertEqual(example["num_ignored"], 1)
            self.assertEqual(example["num_negative"], 1)


if __name__ == "__main__":
    unittest.main()
