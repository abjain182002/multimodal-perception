#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion.clocs_dataset import (
    CONTENT_SHA256_FIELD,
    TRAINING_EXAMPLE_HASH_FIELDS,
    build_training_example,
    load_training_example_npz,
    save_training_example_npz,
    training_example_content_sha256,
)


def make_example():
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
            [0, 1, 2, 2],
            dtype=np.int64,
        ),
        candidate_scores_3d=np.array(
            [0.7, 0.4, 0.6],
            dtype=np.float32,
        ),
        best_iou=np.array(
            [0.82, 0.20, 0.61],
            dtype=np.float32,
        ),
    )


class TestCLOCsDatasetSerialization(unittest.TestCase):

    def test_round_trip_preserves_example(self):
        example = make_example()

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "000001.npz"
            )

            saved = save_training_example_npz(
                example,
                path,
            )

            loaded = load_training_example_npz(
                path
            )

            self.assertEqual(
                loaded[
                    CONTENT_SHA256_FIELD
                ],
                saved[
                    "content_sha256"
                ],
            )

            for key in (
                "schema_version",
                "frame_id",
                "split",
                "num_rows",
                "num_candidates",
                "num_positive",
                "num_negative",
                "num_ignored",
                "coverage_min",
                "coverage_max",
            ):
                self.assertEqual(
                    loaded[key],
                    example[key],
                )

            for key in (
                "feature_rows",
                "candidate_indices",
                "candidate_scores_3d",
                "best_iou",
                "labels",
                "targets",
            ):
                np.testing.assert_array_equal(
                    loaded[key],
                    example[key],
                )

    def test_embedded_hash_matches_canonical_hash(self):
        example = make_example()

        expected = (
            training_example_content_sha256(
                example
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "000001.npz"
            )

            save_training_example_npz(
                example,
                path,
            )

            loaded = load_training_example_npz(
                path
            )

            self.assertEqual(
                loaded[
                    CONTENT_SHA256_FIELD
                ],
                expected,
            )

    def test_repeated_saves_have_same_content_hash(self):
        example = make_example()

        with tempfile.TemporaryDirectory() as tmp:
            path_a = (
                Path(tmp)
                / "a.npz"
            )

            path_b = (
                Path(tmp)
                / "b.npz"
            )

            saved_a = save_training_example_npz(
                example,
                path_a,
            )

            saved_b = save_training_example_npz(
                example,
                path_b,
            )

            self.assertEqual(
                saved_a[
                    "content_sha256"
                ],
                saved_b[
                    "content_sha256"
                ],
            )

    def test_tampered_content_is_detected(self):
        example = make_example()

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "original.npz"
            )

            tampered_path = (
                Path(tmp)
                / "tampered.npz"
            )

            save_training_example_npz(
                example,
                path,
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                payload = {
                    key: np.array(
                        archive[key],
                        copy=True,
                    )
                    for key in archive.files
                }

            payload[
                "candidate_scores_3d"
            ][0] += np.float32(
                0.01
            )

            np.savez(
                tampered_path,
                **payload,
            )

            with self.assertRaisesRegex(
                ValueError,
                "SHA-256 mismatch",
            ):
                load_training_example_npz(
                    tampered_path
                )

    def test_missing_hash_field_is_rejected(self):
        example = make_example()

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "missing_hash.npz"
            )

            payload = {
                key: example[key]
                for key in TRAINING_EXAMPLE_HASH_FIELDS
            }

            np.savez(
                path,
                **payload,
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required field",
            ):
                load_training_example_npz(
                    path
                )

    def test_corrupt_derived_metadata_is_rejected(self):
        example = make_example()

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "original.npz"
            )

            corrupt_path = (
                Path(tmp)
                / "corrupt.npz"
            )

            save_training_example_npz(
                example,
                path,
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                payload = {
                    key: np.array(
                        archive[key],
                        copy=True,
                    )
                    for key in archive.files
                }

            payload[
                "num_positive"
            ] = np.asarray(
                999,
                dtype=np.int64,
            )

            np.savez(
                corrupt_path,
                **payload,
            )

            with self.assertRaisesRegex(
                ValueError,
                "does not match reconstructed value",
            ):
                load_training_example_npz(
                    corrupt_path
                )


if __name__ == "__main__":
    unittest.main()
