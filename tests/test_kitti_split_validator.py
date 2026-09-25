#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_kitti_split import (
    sha256_file,
    validate_split,
)


def write_ids(
    path,
    ids,
):
    Path(path).write_text(
        "".join(
            f"{frame_id}\n"
            for frame_id in ids
        ),
        encoding="utf-8",
    )


def build_manifest(
    train_path,
    val_path,
):
    train_ids = [
        line.strip()
        for line in Path(
            train_path
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    val_ids = [
        line.strip()
        for line in Path(
            val_path
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    return {
        "train": {
            "count":
                len(
                    train_ids
                ),

            "unique_count":
                len(
                    train_set
                ),

            "sha256":
                sha256_file(
                    train_path
                ),
        },

        "validation": {
            "count":
                len(
                    val_ids
                ),

            "unique_count":
                len(
                    val_set
                ),

            "sha256":
                sha256_file(
                    val_path
                ),
        },

        "overlap_count":
            len(
                train_set
                & val_set
            ),

        "union_count":
            len(
                train_set
                | val_set
            ),
    }


class TestKITTISplitValidator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

        self.root = Path(
            self.tmp.name
        )

        self.train_path = (
            self.root
            / "train.txt"
        )

        self.val_path = (
            self.root
            / "val.txt"
        )

        self.manifest_path = (
            self.root
            / "manifest.json"
        )

        write_ids(
            self.train_path,
            [
                "000001",
                "000002",
                "000003",
            ],
        )

        write_ids(
            self.val_path,
            [
                "000004",
                "000005",
            ],
        )

        manifest = build_manifest(
            self.train_path,
            self.val_path,
        )

        self.manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_split_passes(self):
        result = validate_split(
            self.manifest_path,
            self.train_path,
            self.val_path,
        )

        self.assertTrue(
            result["ok"]
        )

        self.assertEqual(
            result["mismatches"],
            {},
        )

    def test_modified_train_file_is_detected(self):
        write_ids(
            self.train_path,
            [
                "000001",
                "000002",
                "999999",
            ],
        )

        result = validate_split(
            self.manifest_path,
            self.train_path,
            self.val_path,
        )

        self.assertFalse(
            result["ok"]
        )

        self.assertIn(
            "train_sha256",
            result["mismatches"],
        )

    def test_duplicate_train_id_is_detected(self):
        write_ids(
            self.train_path,
            [
                "000001",
                "000002",
                "000003",
                "000003",
            ],
        )

        result = validate_split(
            self.manifest_path,
            self.train_path,
            self.val_path,
        )

        self.assertFalse(
            result["ok"]
        )

        self.assertIn(
            "train_count",
            result["mismatches"],
        )

        self.assertIn(
            "train_sha256",
            result["mismatches"],
        )

    def test_train_validation_overlap_is_detected(self):
        write_ids(
            self.val_path,
            [
                "000003",
                "000005",
            ],
        )

        result = validate_split(
            self.manifest_path,
            self.train_path,
            self.val_path,
        )

        self.assertFalse(
            result["ok"]
        )

        self.assertIn(
            "overlap_count",
            result["mismatches"],
        )

        self.assertIn(
            "union_count",
            result["mismatches"],
        )

        self.assertEqual(
            result[
                "observed"
            ][
                "overlap_count"
            ],
            1,
        )

    def test_wrong_manifest_hash_is_detected(self):
        manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        manifest[
            "train"
        ][
            "sha256"
        ] = "0" * 64

        self.manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        result = validate_split(
            self.manifest_path,
            self.train_path,
            self.val_path,
        )

        self.assertFalse(
            result["ok"]
        )

        self.assertIn(
            "train_sha256",
            result["mismatches"],
        )

    def test_missing_split_file_raises(self):
        missing = (
            self.root
            / "missing.txt"
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            validate_split(
                self.manifest_path,
                missing,
                self.val_path,
            )


if __name__ == "__main__":
    unittest.main()
