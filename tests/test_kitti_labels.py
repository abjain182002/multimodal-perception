#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion.kitti_labels import (
    extract_class_camera_boxes,
    parse_kitti_label_file,
    parse_kitti_label_line,
)


class TestKITTILabels(unittest.TestCase):

    def test_parse_ground_truth_line(self):
        line = (
            "Car 0.00 0 1.85 "
            "387.63 181.54 423.81 203.12 "
            "1.67 1.87 3.69 "
            "-16.53 2.39 58.49 1.57"
        )

        record = parse_kitti_label_line(
            line
        )

        self.assertEqual(
            record["name"],
            "Car",
        )

        self.assertIsNone(
            record["score"]
        )

        np.testing.assert_allclose(
            record["dimensions_hwl"],
            np.array(
                [
                    1.67,
                    1.87,
                    3.69,
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            record["camera_box_lhw"],
            np.array(
                [
                    -16.53,
                    2.39,
                    58.49,
                    3.69,
                    1.67,
                    1.87,
                    1.57,
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_parse_result_line_with_score(self):
        line = (
            "Car 0.00 0 1.85 "
            "387.63 181.54 423.81 203.12 "
            "1.67 1.87 3.69 "
            "-16.53 2.39 58.49 1.57 "
            "0.923"
        )

        record = parse_kitti_label_line(
            line
        )

        self.assertAlmostEqual(
            record["score"],
            0.923,
            places=6,
        )

    def test_extract_car_only(self):
        records = [
            parse_kitti_label_line(
                "Truck 0 0 0 "
                "0 0 10 10 "
                "2 2 5 "
                "0 1 20 0"
            ),
            parse_kitti_label_line(
                "Car 0 0 0 "
                "0 0 10 10 "
                "1.5 1.8 4.0 "
                "1 2 30 0.5"
            ),
            parse_kitti_label_line(
                "Cyclist 0 0 0 "
                "0 0 10 10 "
                "1.7 0.5 1.8 "
                "2 1 25 -0.5"
            ),
        ]

        boxes = extract_class_camera_boxes(
            records,
            target_class="Car",
        )

        self.assertEqual(
            boxes.shape,
            (1, 7),
        )

        np.testing.assert_allclose(
            boxes[0],
            np.array(
                [
                    1.0,
                    2.0,
                    30.0,
                    4.0,
                    1.5,
                    1.8,
                    0.5,
                ],
                dtype=np.float32,
            ),
        )

    def test_no_matching_class_returns_empty_shape(self):
        records = [
            parse_kitti_label_line(
                "Cyclist 0 0 0 "
                "0 0 10 10 "
                "1.7 0.5 1.8 "
                "2 1 25 -0.5"
            )
        ]

        boxes = extract_class_camera_boxes(
            records,
            target_class="Car",
        )

        self.assertEqual(
            boxes.shape,
            (0, 7),
        )

        self.assertEqual(
            boxes.dtype,
            np.float32,
        )

    def test_empty_line_raises(self):
        with self.assertRaises(
            ValueError
        ):
            parse_kitti_label_line(
                ""
            )

    def test_wrong_field_count_raises(self):
        with self.assertRaises(
            ValueError
        ):
            parse_kitti_label_line(
                "Car 0 0"
            )

    def test_invalid_numeric_field_raises(self):
        line = (
            "Car 0 0 BAD "
            "0 0 10 10 "
            "1.5 1.8 4.0 "
            "1 2 30 0.5"
        )

        with self.assertRaises(
            ValueError
        ):
            parse_kitti_label_line(
                line
            )

    def test_nonfinite_numeric_field_raises(self):
        line = (
            "Car 0 0 0 "
            "0 0 10 10 "
            "1.5 1.8 4.0 "
            "1 2 nan 0.5"
        )

        with self.assertRaises(
            ValueError
        ):
            parse_kitti_label_line(
                line
            )

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "000000.txt"
            )

            path.write_text(
                "",
                encoding="utf-8",
            )

            records = parse_kitti_label_file(
                path
            )

            self.assertEqual(
                records,
                [],
            )

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "missing.txt"
            )

            with self.assertRaises(
                FileNotFoundError
            ):
                parse_kitti_label_file(
                    path
                )

    def test_empty_target_class_raises(self):
        with self.assertRaises(
            ValueError
        ):
            extract_class_camera_boxes(
                [],
                target_class="",
            )


if __name__ == "__main__":
    unittest.main()
