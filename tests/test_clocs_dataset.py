#!/usr/bin/env python3

import unittest

import numpy as np

from fusion.clocs_dataset import (
    CLOCS_FEATURE_DIM,
    CLOCS_TRAINING_EXAMPLE_SCHEMA,
    build_training_example,
)


def make_valid_example():
    return build_training_example(
        frame_id="000001",
        split="train",
        feature_rows=np.array(
            [
                [0.8, 0.7, 0.9, 0.2],
                [-10.0, 0.4, -10.0, 0.3],
                [0.55, 0.6, 0.75, 0.4],
                [0.30, 0.6, 0.50, 0.4],
            ],
            dtype=np.float32,
        ),
        candidate_indices=np.array(
            [
                0,
                1,
                2,
                2,
            ],
            dtype=np.int64,
        ),
        candidate_scores_3d=np.array(
            [
                0.7,
                0.4,
                0.6,
            ],
            dtype=np.float32,
        ),
        best_iou=np.array(
            [
                0.82,
                0.20,
                0.61,
            ],
            dtype=np.float32,
        ),
    )


class TestCLOCsDataset(unittest.TestCase):

    def test_schema_metadata(self):
        example = make_valid_example()

        self.assertEqual(
            example["schema_version"],
            CLOCS_TRAINING_EXAMPLE_SCHEMA,
        )

        self.assertEqual(
            CLOCS_FEATURE_DIM,
            4,
        )

        self.assertEqual(
            example["frame_id"],
            "000001",
        )

        self.assertEqual(
            example["split"],
            "train",
        )

    def test_shapes_and_dtypes(self):
        example = make_valid_example()

        self.assertEqual(
            example["feature_rows"].shape,
            (4, 4),
        )

        self.assertEqual(
            example["candidate_indices"].shape,
            (4,),
        )

        self.assertEqual(
            example["candidate_scores_3d"].shape,
            (3,),
        )

        self.assertEqual(
            example["best_iou"].shape,
            (3,),
        )

        self.assertEqual(
            example["feature_rows"].dtype,
            np.float32,
        )

        self.assertEqual(
            example["candidate_indices"].dtype,
            np.int64,
        )

        self.assertEqual(
            example["candidate_scores_3d"].dtype,
            np.float32,
        )

    def test_target_semantics(self):
        example = make_valid_example()

        self.assertEqual(
            example["labels"].tolist(),
            [
                1,
                0,
                -1,
            ],
        )

        self.assertEqual(
            example["targets"].tolist(),
            [
                1.0,
                0.0,
                0.0,
            ],
        )

        self.assertEqual(
            example["num_positive"],
            1,
        )

        self.assertEqual(
            example["num_negative"],
            1,
        )

        self.assertEqual(
            example["num_ignored"],
            1,
        )

    def test_multiple_rows_per_candidate_are_allowed(self):
        example = make_valid_example()

        self.assertEqual(
            example["num_rows"],
            4,
        )

        self.assertEqual(
            example["num_candidates"],
            3,
        )

        self.assertEqual(
            example["coverage_min"],
            1,
        )

        self.assertEqual(
            example["coverage_max"],
            2,
        )

    def test_missing_candidate_coverage_raises(self):
        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="000001",
                split="train",
                feature_rows=np.array(
                    [
                        [0.8, 0.7, 0.9, 0.2],
                        [0.4, 0.3, 0.2, 0.1],
                    ],
                    dtype=np.float32,
                ),
                candidate_indices=np.array(
                    [
                        0,
                        0,
                    ],
                    dtype=np.int64,
                ),
                candidate_scores_3d=np.array(
                    [
                        0.7,
                        0.4,
                    ],
                    dtype=np.float32,
                ),
                best_iou=np.array(
                    [
                        0.8,
                        0.2,
                    ],
                    dtype=np.float32,
                ),
            )

    def test_out_of_range_candidate_index_raises(self):
        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="000001",
                split="train",
                feature_rows=np.array(
                    [
                        [0.8, 0.7, 0.9, 0.2],
                        [0.4, 0.3, 0.2, 0.1],
                    ],
                    dtype=np.float32,
                ),
                candidate_indices=np.array(
                    [
                        0,
                        2,
                    ],
                    dtype=np.int64,
                ),
                candidate_scores_3d=np.array(
                    [
                        0.7,
                        0.4,
                    ],
                    dtype=np.float32,
                ),
                best_iou=np.array(
                    [
                        0.8,
                        0.2,
                    ],
                    dtype=np.float32,
                ),
            )

    def test_row_index_length_mismatch_raises(self):
        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="000001",
                split="train",
                feature_rows=np.zeros(
                    (3, 4),
                    dtype=np.float32,
                ),
                candidate_indices=np.array(
                    [
                        0,
                        1,
                    ],
                    dtype=np.int64,
                ),
                candidate_scores_3d=np.array(
                    [
                        0.7,
                        0.4,
                    ],
                    dtype=np.float32,
                ),
                best_iou=np.array(
                    [
                        0.8,
                        0.2,
                    ],
                    dtype=np.float32,
                ),
            )

    def test_invalid_feature_shape_raises(self):
        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="000001",
                split="train",
                feature_rows=np.zeros(
                    (2, 5),
                    dtype=np.float32,
                ),
                candidate_indices=np.array(
                    [
                        0,
                        1,
                    ],
                    dtype=np.int64,
                ),
                candidate_scores_3d=np.array(
                    [
                        0.7,
                        0.4,
                    ],
                    dtype=np.float32,
                ),
                best_iou=np.array(
                    [
                        0.8,
                        0.2,
                    ],
                    dtype=np.float32,
                ),
            )

    def test_invalid_candidate_scores_raise(self):
        invalid_sets = [
            [-0.1, 0.5],
            [0.5, 1.1],
            [0.5, np.nan],
            [0.5, np.inf],
        ]

        for scores in invalid_sets:
            with self.subTest(
                scores=scores
            ):
                with self.assertRaises(
                    ValueError
                ):
                    build_training_example(
                        frame_id="000001",
                        split="train",
                        feature_rows=np.zeros(
                            (2, 4),
                            dtype=np.float32,
                        ),
                        candidate_indices=np.array(
                            [
                                0,
                                1,
                            ],
                            dtype=np.int64,
                        ),
                        candidate_scores_3d=np.array(
                            scores,
                            dtype=np.float32,
                        ),
                        best_iou=np.array(
                            [
                                0.8,
                                0.2,
                            ],
                            dtype=np.float32,
                        ),
                    )

    def test_empty_identifiers_raise(self):
        kwargs = dict(
            feature_rows=np.zeros(
                (2, 4),
                dtype=np.float32,
            ),
            candidate_indices=np.array(
                [
                    0,
                    1,
                ],
                dtype=np.int64,
            ),
            candidate_scores_3d=np.array(
                [
                    0.7,
                    0.4,
                ],
                dtype=np.float32,
            ),
            best_iou=np.array(
                [
                    0.8,
                    0.2,
                ],
                dtype=np.float32,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="",
                split="train",
                **kwargs,
            )

        with self.assertRaises(
            ValueError
        ):
            build_training_example(
                frame_id="000001",
                split="",
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
