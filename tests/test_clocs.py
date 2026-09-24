#!/usr/bin/env python3

import unittest

import numpy as np

from fusion.clocs import (
    build_clocs_feature_rows,
    normalized_lidar_distance,
    pairwise_iou_xyxy,
)


class TestCLOCsFeatureConstruction(unittest.TestCase):

    def setUp(self):
        self.projected_boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [5, 5, 15, 15],
        ], dtype=np.float32)

        self.scores_3d = np.array([
            0.9,
            0.2,
            0.5,
        ], dtype=np.float32)

        self.boxes_2d = np.array([
            [0, 0, 10, 10],
            [8, 8, 12, 12],
        ], dtype=np.float32)

        self.scores_2d = np.array([
            0.8,
            0.6,
        ], dtype=np.float32)

        self.distance_82 = np.array([
            0.1,
            0.2,
            0.3,
        ], dtype=np.float32)

    def test_pairwise_iou(self):
        ious = pairwise_iou_xyxy(
            self.projected_boxes,
            self.boxes_2d,
        )

        self.assertEqual(
            ious.shape,
            (3, 2),
        )

        np.testing.assert_allclose(
            ious[0, 0],
            1.0,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            ious[0, 1],
            4.0 / 112.0,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            ious[2, 0],
            25.0 / 175.0,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            ious[2, 1],
            0.16,
            rtol=1e-6,
        )

    def test_feature_rows_and_ordering(self):
        result = build_clocs_feature_rows(
            projected_boxes=self.projected_boxes,
            scores_3d=self.scores_3d,
            boxes_2d=self.boxes_2d,
            scores_2d=self.scores_2d,
            distance_82=self.distance_82,
        )

        self.assertEqual(
            result["features"].shape,
            (5, 4),
        )

        self.assertEqual(
            result["candidate_index"].tolist(),
            [0, 0, 1, 2, 2],
        )

        self.assertEqual(
            result["detector_2d_index"].tolist(),
            [0, 1, -1, 0, 1],
        )

        self.assertEqual(
            result["row_type"].tolist(),
            [1, 1, 0, 1, 1],
        )

    def test_unmatched_candidate_encoding(self):
        result = build_clocs_feature_rows(
            projected_boxes=self.projected_boxes,
            scores_3d=self.scores_3d,
            boxes_2d=self.boxes_2d,
            scores_2d=self.scores_2d,
            distance_82=self.distance_82,
        )

        np.testing.assert_allclose(
            result["features"][2],
            np.array([
                -10.0,
                0.2,
                -10.0,
                0.2,
            ], dtype=np.float32),
            rtol=1e-6,
        )

    def test_no_2d_detections(self):
        result = build_clocs_feature_rows(
            projected_boxes=self.projected_boxes,
            scores_3d=self.scores_3d,
            boxes_2d=np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            scores_2d=np.empty(
                (0,),
                dtype=np.float32,
            ),
            distance_82=self.distance_82,
        )

        self.assertEqual(
            result["features"].shape,
            (3, 4),
        )

        self.assertEqual(
            result["detector_2d_index"].tolist(),
            [-1, -1, -1],
        )

        self.assertEqual(
            result["row_type"].tolist(),
            [0, 0, 0],
        )

        self.assertTrue(
            np.all(
                result["features"][:, 0]
                == -10.0
            )
        )

        self.assertTrue(
            np.all(
                result["features"][:, 2]
                == -10.0
            )
        )

    def test_normalized_lidar_distance(self):
        lidar_boxes = np.array([
            [3.0, 4.0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        distance = normalized_lidar_distance(
            lidar_boxes
        )

        np.testing.assert_allclose(
            distance[0],
            5.0 / 82.0,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
