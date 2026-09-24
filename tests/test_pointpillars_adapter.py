#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import (
    build_kitti_lidar2cam,
    load_kitti_calibration,
)


class TestPointPillarsAdapter(unittest.TestCase):

    def test_load_kitti_calibration(self):
        calibration_text = """\
P2: 7.0 0.0 6.0 4.0 0.0 7.0 2.0 3.0 0.0 0.0 1.0 0.0
R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
Tr_velo_to_cam: 1.0 0.0 0.0 1.0 0.0 1.0 0.0 2.0 0.0 0.0 1.0 3.0
"""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000001.txt"
            path.write_text(
                calibration_text,
                encoding="utf-8",
            )

            calibration = load_kitti_calibration(
                path
            )

        self.assertEqual(
            calibration["P2"].shape,
            (12,),
        )

        self.assertEqual(
            calibration["R0_rect"].shape,
            (9,),
        )

        self.assertEqual(
            calibration["Tr_velo_to_cam"].shape,
            (12,),
        )

        self.assertEqual(
            calibration["P2"].dtype,
            np.float32,
        )

    def test_build_kitti_lidar2cam_identity_rectification(self):
        calibration = {
            "R0_rect": np.eye(
                3,
                dtype=np.float32,
            ).reshape(-1),

            "Tr_velo_to_cam": np.array([
                1.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 2.0,
                0.0, 0.0, 1.0, 3.0,
            ], dtype=np.float32),

            "P2": np.zeros(
                12,
                dtype=np.float32,
            ),
        }

        lidar2cam = build_kitti_lidar2cam(
            calibration
        )

        expected = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(
            lidar2cam,
            expected,
        )

    def test_build_kitti_lidar2cam_applies_rectification(self):
        calibration = {
            "R0_rect": np.array([
                0.0, -1.0, 0.0,
                1.0,  0.0, 0.0,
                0.0,  0.0, 1.0,
            ], dtype=np.float32),

            "Tr_velo_to_cam": np.array([
                1.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 2.0,
                0.0, 0.0, 1.0, 3.0,
            ], dtype=np.float32),

            "P2": np.zeros(
                12,
                dtype=np.float32,
            ),
        }

        lidar2cam = build_kitti_lidar2cam(
            calibration
        )

        expected = np.array([
            [0.0, -1.0, 0.0, -2.0],
            [1.0,  0.0, 0.0,  1.0],
            [0.0,  0.0, 1.0,  3.0],
            [0.0,  0.0, 0.0,  1.0],
        ], dtype=np.float32)

        np.testing.assert_allclose(
            lidar2cam,
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_missing_required_calibration_field(self):
        calibration_text = """\
P2: 7.0 0.0 6.0 4.0 0.0 7.0 2.0 3.0 0.0 0.0 1.0 0.0
R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
"""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.txt"

            path.write_text(
                calibration_text,
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                load_kitti_calibration(
                    path
                )


if __name__ == "__main__":
    unittest.main()
