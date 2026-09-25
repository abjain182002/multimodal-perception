#!/usr/bin/env python3

import unittest

import numpy as np

from fusion.clocs_training import (
    IGNORE_LABEL,
    NEGATIVE_IOU_THRESHOLD,
    NEGATIVE_LABEL,
    POSITIVE_IOU_THRESHOLD,
    POSITIVE_LABEL,
    build_candidate_targets,
)


class TestCLOCsTrainingTargets(unittest.TestCase):

    def test_exact_threshold_boundaries(self):
        result = build_candidate_targets(
            [
                0.0,
                0.50,
                0.500001,
                0.699999,
                0.70,
                1.0,
            ]
        )

        self.assertEqual(
            result["labels"].tolist(),
            [
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
                IGNORE_LABEL,
                IGNORE_LABEL,
                POSITIVE_LABEL,
                POSITIVE_LABEL,
            ],
        )

        self.assertEqual(
            result["targets"].tolist(),
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        )

    def test_masks_and_counts_are_consistent(self):
        result = build_candidate_targets(
            [
                0.10,
                0.40,
                0.55,
                0.65,
                0.75,
                0.90,
            ]
        )

        self.assertEqual(
            result["num_positive"],
            2,
        )

        self.assertEqual(
            result["num_negative"],
            2,
        )

        self.assertEqual(
            result["num_ignored"],
            2,
        )

        combined = (
            result["positive_mask"].astype(
                np.int8
            )
            + result["negative_mask"].astype(
                np.int8
            )
            + result["ignore_mask"].astype(
                np.int8
            )
        )

        np.testing.assert_array_equal(
            combined,
            np.ones(
                6,
                dtype=np.int8,
            ),
        )

    def test_no_ground_truth_semantics(self):
        """
        A frame with candidates but no Car GT has best IoU zero
        for every candidate, so every candidate is negative.
        """

        result = build_candidate_targets(
            np.zeros(
                5,
                dtype=np.float32,
            )
        )

        self.assertEqual(
            result["labels"].tolist(),
            [
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
            ],
        )

        self.assertEqual(
            result["num_positive"],
            0,
        )

        self.assertEqual(
            result["num_negative"],
            5,
        )

        self.assertEqual(
            result["num_ignored"],
            0,
        )

    def test_ignored_candidates_have_zero_binary_target(self):
        result = build_candidate_targets(
            [
                0.55,
                0.60,
                0.65,
            ]
        )

        np.testing.assert_array_equal(
            result["labels"],
            np.full(
                3,
                IGNORE_LABEL,
                dtype=np.int8,
            ),
        )

        np.testing.assert_array_equal(
            result["targets"],
            np.zeros(
                3,
                dtype=np.float32,
            ),
        )

    def test_output_dtypes(self):
        result = build_candidate_targets(
            [
                0.20,
                0.60,
                0.80,
            ]
        )

        self.assertEqual(
            result["best_iou"].dtype,
            np.float32,
        )

        self.assertEqual(
            result["labels"].dtype,
            np.int8,
        )

        self.assertEqual(
            result["targets"].dtype,
            np.float32,
        )

        self.assertEqual(
            result["positive_mask"].dtype,
            np.bool_,
        )

        self.assertEqual(
            result["negative_mask"].dtype,
            np.bool_,
        )

        self.assertEqual(
            result["ignore_mask"].dtype,
            np.bool_,
        )

    def test_custom_thresholds(self):
        result = build_candidate_targets(
            [
                0.10,
                0.20,
                0.40,
                0.60,
                0.80,
            ],
            positive_iou_threshold=0.60,
            negative_iou_threshold=0.20,
        )

        self.assertEqual(
            result["labels"].tolist(),
            [
                NEGATIVE_LABEL,
                NEGATIVE_LABEL,
                IGNORE_LABEL,
                POSITIVE_LABEL,
                POSITIVE_LABEL,
            ],
        )

    def test_invalid_threshold_order_raises(self):
        for negative, positive in [
            (0.70, 0.70),
            (0.80, 0.70),
        ]:
            with self.subTest(
                negative=negative,
                positive=positive,
            ):
                with self.assertRaises(
                    ValueError
                ):
                    build_candidate_targets(
                        [0.5],
                        positive_iou_threshold=positive,
                        negative_iou_threshold=negative,
                    )

    def test_invalid_iou_values_raise(self):
        invalid_inputs = [
            [-0.01],
            [1.01],
            [np.nan],
            [np.inf],
            [-np.inf],
        ]

        for values in invalid_inputs:
            with self.subTest(
                values=values
            ):
                with self.assertRaises(
                    ValueError
                ):
                    build_candidate_targets(
                        values
                    )

    def test_default_threshold_constants(self):
        self.assertEqual(
            POSITIVE_IOU_THRESHOLD,
            0.70,
        )

        self.assertEqual(
            NEGATIVE_IOU_THRESHOLD,
            0.50,
        )


if __name__ == "__main__":
    unittest.main()
