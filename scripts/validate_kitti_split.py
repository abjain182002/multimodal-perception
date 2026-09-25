#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_frame_ids(path):
    return [
        line.strip()
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def validate_split(
    manifest_path,
    train_path,
    val_path,
):
    manifest_path = Path(
        manifest_path
    )

    train_path = Path(
        train_path
    )

    val_path = Path(
        val_path
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Train split not found: {train_path}"
        )

    if not val_path.is_file():
        raise FileNotFoundError(
            f"Validation split not found: {val_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    train_ids = read_frame_ids(
        train_path
    )

    val_ids = read_frame_ids(
        val_path
    )

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    overlap = (
        train_set
        & val_set
    )

    union = (
        train_set
        | val_set
    )

    observed = {
        "train_count":
            len(
                train_ids
            ),

        "train_unique_count":
            len(
                train_set
            ),

        "train_sha256":
            sha256_file(
                train_path
            ),

        "validation_count":
            len(
                val_ids
            ),

        "validation_unique_count":
            len(
                val_set
            ),

        "validation_sha256":
            sha256_file(
                val_path
            ),

        "overlap_count":
            len(
                overlap
            ),

        "union_count":
            len(
                union
            ),
    }

    expected = {
        "train_count":
            int(
                manifest[
                    "train"
                ][
                    "count"
                ]
            ),

        "train_unique_count":
            int(
                manifest[
                    "train"
                ][
                    "unique_count"
                ]
            ),

        "train_sha256":
            str(
                manifest[
                    "train"
                ][
                    "sha256"
                ]
            ),

        "validation_count":
            int(
                manifest[
                    "validation"
                ][
                    "count"
                ]
            ),

        "validation_unique_count":
            int(
                manifest[
                    "validation"
                ][
                    "unique_count"
                ]
            ),

        "validation_sha256":
            str(
                manifest[
                    "validation"
                ][
                    "sha256"
                ]
            ),

        "overlap_count":
            int(
                manifest[
                    "overlap_count"
                ]
            ),

        "union_count":
            int(
                manifest[
                    "union_count"
                ]
            ),
    }

    mismatches = {}

    for key in expected:
        if observed[
            key
        ] != expected[
            key
        ]:
            mismatches[
                key
            ] = {
                "expected":
                    expected[
                        key
                    ],

                "observed":
                    observed[
                        key
                    ],
            }

    return {
        "ok":
            len(
                mismatches
            ) == 0,

        "observed":
            observed,

        "expected":
            expected,

        "mismatches":
            mismatches,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate KITTI train/validation split files "
            "against the frozen experiment manifest."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to kitti_second_split.json",
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to KITTI train.txt",
    )

    parser.add_argument(
        "--val",
        required=True,
        help="Path to KITTI val.txt",
    )

    args = parser.parse_args()

    result = validate_split(
        manifest_path=args.manifest,
        train_path=args.train,
        val_path=args.val,
    )

    print(
        "=== KITTI SPLIT VALIDATION ==="
    )

    for key, value in result[
        "observed"
    ].items():
        print(
            f"{key:26s}: {value}"
        )

    print()

    if result[
        "ok"
    ]:
        print(
            "KITTI SPLIT VALIDATION: PASS"
        )

        return 0

    print(
        "KITTI SPLIT VALIDATION: FAIL"
    )

    print()

    for key, mismatch in result[
        "mismatches"
    ].items():
        print(
            f"{key}:"
        )

        print(
            "  expected:",
            mismatch[
                "expected"
            ],
        )

        print(
            "  observed:",
            mismatch[
                "observed"
            ],
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
