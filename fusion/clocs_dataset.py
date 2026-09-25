#!/usr/bin/env python3

import numpy as np

from fusion.clocs_training import (
    build_candidate_targets,
)


CLOCS_TRAINING_EXAMPLE_SCHEMA = "clocs_training_example_v1"
CLOCS_FEATURE_DIM = 4


def build_training_example(
    frame_id,
    split,
    feature_rows,
    candidate_indices,
    candidate_scores_3d,
    best_iou,
):
    """
    Build and validate one CLOCs per-frame training example.

    Parameters
    ----------
    frame_id:
        KITTI frame identifier.

    split:
        Dataset split label, e.g. "train" or "val".

    feature_rows:
        Candidate-level CLOCs feature rows with shape [R, 4].

    candidate_indices:
        PointPillars candidate index corresponding to each
        feature row, shape [R].

    candidate_scores_3d:
        Frozen PointPillars Car score for every dense candidate,
        shape [N].

    best_iou:
        Maximum 3D IoU of every candidate against Car ground
        truth for the frame, shape [N].

    Returns
    -------
    dict
        Validated training example.
    """

    frame_id = str(frame_id).strip()
    split = str(split).strip()

    if not frame_id:
        raise ValueError(
            "frame_id cannot be empty"
        )

    if not split:
        raise ValueError(
            "split cannot be empty"
        )

    feature_rows = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=np.int64,
    ).reshape(-1)

    candidate_scores_3d = np.asarray(
        candidate_scores_3d,
        dtype=np.float32,
    ).reshape(-1)

    best_iou = np.asarray(
        best_iou,
        dtype=np.float32,
    ).reshape(-1)

    if (
        feature_rows.ndim != 2
        or feature_rows.shape[1] != CLOCS_FEATURE_DIM
    ):
        raise ValueError(
            "feature_rows must have shape [R, 4]"
        )

    if feature_rows.shape[0] == 0:
        raise ValueError(
            "feature_rows cannot be empty"
        )

    if not np.isfinite(
        feature_rows
    ).all():
        raise ValueError(
            "feature_rows contain non-finite values"
        )

    if (
        candidate_indices.shape[0]
        != feature_rows.shape[0]
    ):
        raise ValueError(
            "candidate_indices must have one entry "
            "per feature row"
        )

    num_candidates = int(
        candidate_scores_3d.shape[0]
    )

    if num_candidates == 0:
        raise ValueError(
            "At least one PointPillars candidate is required"
        )

    if best_iou.shape[0] != num_candidates:
        raise ValueError(
            "best_iou must have one entry per candidate"
        )

    if not np.isfinite(
        candidate_scores_3d
    ).all():
        raise ValueError(
            "candidate_scores_3d contains non-finite values"
        )

    if np.any(
        candidate_scores_3d < 0.0
    ) or np.any(
        candidate_scores_3d > 1.0
    ):
        raise ValueError(
            "candidate_scores_3d must be in [0, 1]"
        )

    if np.any(
        candidate_indices < 0
    ) or np.any(
        candidate_indices >= num_candidates
    ):
        raise ValueError(
            "candidate_indices contains an out-of-range index"
        )

    # CLOCs dynamic max aggregation requires every PointPillars
    # candidate to be represented by at least one row.
    coverage = np.bincount(
        candidate_indices,
        minlength=num_candidates,
    )

    if np.any(
        coverage == 0
    ):
        missing = np.flatnonzero(
            coverage == 0
        )

        raise ValueError(
            "Every candidate must have at least one feature row; "
            f"missing {missing.size} candidate(s)"
        )

    target_info = build_candidate_targets(
        best_iou
    )

    example = {
        "schema_version":
            CLOCS_TRAINING_EXAMPLE_SCHEMA,

        "frame_id":
            frame_id,

        "split":
            split,

        "feature_rows":
            feature_rows,

        "candidate_indices":
            candidate_indices,

        "candidate_scores_3d":
            candidate_scores_3d,

        "best_iou":
            target_info["best_iou"],

        "labels":
            target_info["labels"],

        "targets":
            target_info["targets"],

        "num_rows":
            int(
                feature_rows.shape[0]
            ),

        "num_candidates":
            num_candidates,

        "num_positive":
            target_info["num_positive"],

        "num_negative":
            target_info["num_negative"],

        "num_ignored":
            target_info["num_ignored"],

        "coverage_min":
            int(
                coverage.min()
            ),

        "coverage_max":
            int(
                coverage.max()
            ),
    }

    return example


def _canonical_array_bytes(value):
    """
    Convert one training-example value to deterministic NumPy .npy bytes.

    The resulting bytes are used for content hashing and are independent
    of dictionary insertion order.
    """

    import io

    array = np.asarray(
        value
    )

    buffer = io.BytesIO()

    np.lib.format.write_array(
        buffer,
        array,
        allow_pickle=False,
    )

    return buffer.getvalue()


TRAINING_EXAMPLE_HASH_FIELDS = (
    "schema_version",
    "frame_id",
    "split",
    "feature_rows",
    "candidate_indices",
    "candidate_scores_3d",
    "best_iou",
    "labels",
    "targets",
    "num_rows",
    "num_candidates",
    "num_positive",
    "num_negative",
    "num_ignored",
    "coverage_min",
    "coverage_max",
)


def validate_training_example(example):
    """
    Validate a complete CLOCs training-example dictionary.

    Derived targets, counts, and coverage metadata are independently
    reconstructed from the primitive stored values and must agree exactly.
    """

    if not isinstance(
        example,
        dict,
    ):
        raise TypeError(
            "example must be a dictionary"
        )

    missing = [
        key
        for key in TRAINING_EXAMPLE_HASH_FIELDS
        if key not in example
    ]

    if missing:
        raise ValueError(
            "training example is missing required field(s): "
            + ", ".join(
                missing
            )
        )

    rebuilt = build_training_example(
        frame_id=example[
            "frame_id"
        ],
        split=example[
            "split"
        ],
        feature_rows=example[
            "feature_rows"
        ],
        candidate_indices=example[
            "candidate_indices"
        ],
        candidate_scores_3d=example[
            "candidate_scores_3d"
        ],
        best_iou=example[
            "best_iou"
        ],
    )

    scalar_fields = (
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
    )

    array_fields = (
        "feature_rows",
        "candidate_indices",
        "candidate_scores_3d",
        "best_iou",
        "labels",
        "targets",
    )

    for key in scalar_fields:
        if example[
            key
        ] != rebuilt[
            key
        ]:
            raise ValueError(
                f"training-example field '{key}' "
                "does not match reconstructed value"
            )

    for key in array_fields:
        supplied = np.asarray(
            example[
                key
            ]
        )

        expected = np.asarray(
            rebuilt[
                key
            ]
        )

        if (
            supplied.shape
            != expected.shape
        ):
            raise ValueError(
                f"training-example field '{key}' "
                "has an invalid shape"
            )

        if not np.array_equal(
            supplied,
            expected,
        ):
            raise ValueError(
                f"training-example field '{key}' "
                "does not match reconstructed value"
            )

    return rebuilt


def training_example_content_sha256(
    example,
):
    """
    Compute a deterministic SHA-256 over the canonical example contents.
    """

    import hashlib

    canonical = validate_training_example(
        example
    )

    digest = hashlib.sha256()

    for key in TRAINING_EXAMPLE_HASH_FIELDS:
        key_bytes = key.encode(
            "utf-8"
        )

        value_bytes = _canonical_array_bytes(
            canonical[
                key
            ]
        )

        digest.update(
            len(
                key_bytes
            ).to_bytes(
                4,
                byteorder="big",
            )
        )

        digest.update(
            key_bytes
        )

        digest.update(
            len(
                value_bytes
            ).to_bytes(
                8,
                byteorder="big",
            )
        )

        digest.update(
            value_bytes
        )

    return digest.hexdigest()


CONTENT_SHA256_FIELD = "content_sha256"


def save_training_example_npz(
    example,
    path,
):
    """
    Save one validated CLOCs training example to an NPZ archive.

    The archive contains an embedded SHA-256 computed from the
    canonical training-example contents.
    """

    from pathlib import Path

    canonical = validate_training_example(
        example
    )

    content_sha256 = (
        training_example_content_sha256(
            canonical
        )
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        key: canonical[
            key
        ]
        for key in TRAINING_EXAMPLE_HASH_FIELDS
    }

    payload[
        CONTENT_SHA256_FIELD
    ] = np.asarray(
        content_sha256,
        dtype="<U64",
    )

    np.savez(
        path,
        **payload,
    )

    return {
        "path": str(
            path
        ),
        "content_sha256": content_sha256,
    }


def _npz_scalar(
    archive,
    key,
):
    value = archive[
        key
    ]

    if value.shape != ():
        raise ValueError(
            f"NPZ field '{key}' must be scalar"
        )

    return value.item()


def load_training_example_npz(
    path,
    verify_integrity=True,
):
    """
    Load one CLOCs training example from NPZ.

    By default, the embedded content SHA-256 is recomputed and
    verified before the example is returned.
    """

    from pathlib import Path

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Training example not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        required_archive_fields = set(
            TRAINING_EXAMPLE_HASH_FIELDS
        ) | {
            CONTENT_SHA256_FIELD
        }

        missing = (
            required_archive_fields
            - set(
                archive.files
            )
        )

        if missing:
            raise ValueError(
                "NPZ archive is missing required field(s): "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        example = {
            "schema_version":
                str(
                    _npz_scalar(
                        archive,
                        "schema_version",
                    )
                ),

            "frame_id":
                str(
                    _npz_scalar(
                        archive,
                        "frame_id",
                    )
                ),

            "split":
                str(
                    _npz_scalar(
                        archive,
                        "split",
                    )
                ),

            "feature_rows":
                np.array(
                    archive[
                        "feature_rows"
                    ],
                    copy=True,
                ),

            "candidate_indices":
                np.array(
                    archive[
                        "candidate_indices"
                    ],
                    copy=True,
                ),

            "candidate_scores_3d":
                np.array(
                    archive[
                        "candidate_scores_3d"
                    ],
                    copy=True,
                ),

            "best_iou":
                np.array(
                    archive[
                        "best_iou"
                    ],
                    copy=True,
                ),

            "labels":
                np.array(
                    archive[
                        "labels"
                    ],
                    copy=True,
                ),

            "targets":
                np.array(
                    archive[
                        "targets"
                    ],
                    copy=True,
                ),

            "num_rows":
                int(
                    _npz_scalar(
                        archive,
                        "num_rows",
                    )
                ),

            "num_candidates":
                int(
                    _npz_scalar(
                        archive,
                        "num_candidates",
                    )
                ),

            "num_positive":
                int(
                    _npz_scalar(
                        archive,
                        "num_positive",
                    )
                ),

            "num_negative":
                int(
                    _npz_scalar(
                        archive,
                        "num_negative",
                    )
                ),

            "num_ignored":
                int(
                    _npz_scalar(
                        archive,
                        "num_ignored",
                    )
                ),

            "coverage_min":
                int(
                    _npz_scalar(
                        archive,
                        "coverage_min",
                    )
                ),

            "coverage_max":
                int(
                    _npz_scalar(
                        archive,
                        "coverage_max",
                    )
                ),
        }

        stored_sha256 = str(
            _npz_scalar(
                archive,
                CONTENT_SHA256_FIELD,
            )
        )

    canonical = validate_training_example(
        example
    )

    recomputed_sha256 = (
        training_example_content_sha256(
            canonical
        )
    )

    if (
        verify_integrity
        and stored_sha256
        != recomputed_sha256
    ):
        raise ValueError(
            "Training-example content SHA-256 mismatch"
        )

    canonical[
        CONTENT_SHA256_FIELD
    ] = stored_sha256

    return canonical
