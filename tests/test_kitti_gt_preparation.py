#!/usr/bin/env python3

import unittest

import numpy as np

from fusion.kitti_labels import (
    camera_boxes_to_clocs_lidar_boxes,
    filter_lidar_boxes_by_bev_range,
    lidar_bev_box_corners,
    prepare_clocs_car_ground_truth,
)


class TestKITTIGroundTruthPreparation(unittest.TestCase):

    def test_identity_transform_reorders_dimensions(self):
        camera_boxes = np.asarray(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,   # length
                    1.5,   # height
                    1.8,   # width
                    0.25,
                ]
            ],
            dtype=np.float32,
        )

        result = camera_boxes_to_clocs_lidar_boxes(
            camera_boxes,
            np.eye(
                4,
                dtype=np.float32,
            ),
        )

        expected = np.asarray(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    1.8,   # width
                    4.0,   # length
                    1.5,   # height
                    0.25,
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result,
            expected,
            rtol=0.0,
            atol=1e-6,
        )

    def test_translation_transform_is_inverted(self):
        lidar_to_camera = np.eye(
            4,
            dtype=np.float32,
        )

        lidar_to_camera[
            0,
            3,
        ] = 10.0

        camera_boxes = np.asarray(
            [
                [
                    12.0,
                    3.0,
                    4.0,
                    4.0,
                    1.5,
                    2.0,
                    0.0,
                ]
            ],
            dtype=np.float32,
        )

        result = camera_boxes_to_clocs_lidar_boxes(
            camera_boxes,
            lidar_to_camera,
        )

        np.testing.assert_allclose(
            result[
                0,
                :3,
            ],
            np.asarray(
                [
                    2.0,
                    3.0,
                    4.0,
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_zero_rotation_bev_corners(self):
        boxes = np.asarray(
            [
                [
                    5.0,
                    6.0,
                    0.0,
                    2.0,   # width
                    4.0,   # length
                    1.5,
                    0.0,
                ]
            ],
            dtype=np.float32,
        )

        corners = lidar_bev_box_corners(
            boxes
        )

        expected = np.asarray(
            [
                [
                    [4.0, 4.0],
                    [4.0, 8.0],
                    [6.0, 8.0],
                    [6.0, 4.0],
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            corners,
            expected,
            rtol=0.0,
            atol=1e-6,
        )

    def test_fully_inside_box_is_kept(self):
        boxes = np.asarray(
            [
                [
                    5.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    1.5,
                    0.0,
                ]
            ],
            dtype=np.float32,
        )

        mask = filter_lidar_boxes_by_bev_range(
            boxes,
            [
                0.0,
                -10.0,
                -3.0,
                10.0,
                10.0,
                1.0,
            ],
        )

        self.assertEqual(
            mask.tolist(),
            [True],
        )

    def test_partially_intersecting_box_is_kept(self):
        # Centre is outside x_max=10, but the box extends
        # into the valid range. CLOCs/SECOND keeps it because
        # at least one rotated BEV corner lies inside.
        boxes = np.asarray(
            [
                [
                    10.5,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                    1.5,
                    0.0,
                ]
            ],
            dtype=np.float32,
        )

        mask = filter_lidar_boxes_by_bev_range(
            boxes,
            [
                0.0,
                -10.0,
                -3.0,
                10.0,
                10.0,
                1.0,
            ],
        )

        self.assertEqual(
            mask.tolist(),
            [True],
        )

    def test_completely_outside_box_is_rejected(self):
        boxes = np.asarray(
            [
                [
                    12.0,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                    1.5,
                    0.0,
                ]
            ],
            dtype=np.float32,
        )

        mask = filter_lidar_boxes_by_bev_range(
            boxes,
            [
                0.0,
                -10.0,
                -3.0,
                10.0,
                10.0,
                1.0,
            ],
        )

        self.assertEqual(
            mask.tolist(),
            [False],
        )

    def test_empty_boxes_are_supported(self):
        empty = np.zeros(
            (0, 7),
            dtype=np.float32,
        )

        converted = (
            camera_boxes_to_clocs_lidar_boxes(
                empty,
                np.eye(
                    4,
                    dtype=np.float32,
                ),
            )
        )

        corners = lidar_bev_box_corners(
            empty
        )

        mask = filter_lidar_boxes_by_bev_range(
            empty,
            [
                0.0,
                -10.0,
                -3.0,
                10.0,
                10.0,
                1.0,
            ],
        )

        self.assertEqual(
            converted.shape,
            (0, 7),
        )

        self.assertEqual(
            corners.shape,
            (0, 4, 2),
        )

        self.assertEqual(
            mask.shape,
            (0,),
        )

    def test_prepare_no_car_ground_truth(self):
        records = [
            {
                "name": "Cyclist",
                "camera_box_lhw": np.asarray(
                    [
                        5.0,
                        0.0,
                        20.0,
                        1.8,
                        1.7,
                        0.5,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            }
        ]

        result = prepare_clocs_car_ground_truth(
            records,
            np.eye(
                4,
                dtype=np.float32,
            ),
            point_cloud_range=[
                0.0,
                -10.0,
                -3.0,
                10.0,
                10.0,
                1.0,
            ],
        )

        self.assertEqual(
            result[
                "num_car_before_range_filter"
            ],
            0,
        )

        self.assertEqual(
            result[
                "num_car_after_range_filter"
            ],
            0,
        )

        self.assertEqual(
            result[
                "camera_boxes"
            ].shape,
            (0, 7),
        )

        self.assertEqual(
            result[
                "lidar_boxes"
            ].shape,
            (0, 7),
        )

    def test_invalid_transform_shape_raises(self):
        with self.assertRaises(
            ValueError
        ):
            camera_boxes_to_clocs_lidar_boxes(
                np.zeros(
                    (1, 7),
                    dtype=np.float32,
                ),
                np.eye(
                    3,
                    dtype=np.float32,
                ),
            )

    def test_invalid_range_raises(self):
        boxes = np.zeros(
            (1, 7),
            dtype=np.float32,
        )

        with self.assertRaises(
            ValueError
        ):
            filter_lidar_boxes_by_bev_range(
                boxes,
                [
                    10.0,
                    -5.0,
                    -3.0,
                    0.0,
                    5.0,
                    1.0,
                ],
            )


if __name__ == "__main__":
    unittest.main()
