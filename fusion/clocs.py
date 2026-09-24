#!/usr/bin/env python3

import numpy as np


UNMATCHED_IOU = -10.0
UNMATCHED_2D_SCORE = -10.0
DISTANCE_NORMALIZER = 82.0


def pairwise_iou_xyxy(boxes_a, boxes_b):
    """
    Compute pairwise 2D IoU for boxes in:
        x1, y1, x2, y2

    Returns an array with shape:
        [num_boxes_a, num_boxes_b]
    """

    boxes_a = np.asarray(
        boxes_a,
        dtype=np.float32,
    )

    boxes_b = np.asarray(
        boxes_b,
        dtype=np.float32,
    )

    if boxes_a.ndim != 2 or boxes_a.shape[1] != 4:
        raise ValueError(
            "boxes_a must have shape [N, 4]"
        )

    if boxes_b.ndim != 2 or boxes_b.shape[1] != 4:
        raise ValueError(
            "boxes_b must have shape [M, 4]"
        )

    num_a = boxes_a.shape[0]
    num_b = boxes_b.shape[0]

    if num_a == 0 or num_b == 0:
        return np.zeros(
            (num_a, num_b),
            dtype=np.float32,
        )

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    x1 = np.maximum(
        a[..., 0],
        b[..., 0],
    )

    y1 = np.maximum(
        a[..., 1],
        b[..., 1],
    )

    x2 = np.minimum(
        a[..., 2],
        b[..., 2],
    )

    y2 = np.minimum(
        a[..., 3],
        b[..., 3],
    )

    width = np.maximum(
        x2 - x1,
        0.0,
    )

    height = np.maximum(
        y2 - y1,
        0.0,
    )

    intersection = width * height

    area_a = np.maximum(
        (a[..., 2] - a[..., 0])
        * (a[..., 3] - a[..., 1]),
        0.0,
    )

    area_b = np.maximum(
        (b[..., 2] - b[..., 0])
        * (b[..., 3] - b[..., 1]),
        0.0,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    iou = np.zeros_like(
        intersection,
        dtype=np.float32,
    )

    np.divide(
        intersection,
        union,
        out=iou,
        where=union > 0,
    )

    return iou


def normalized_lidar_distance(
    lidar_boxes,
    normalizer=DISTANCE_NORMALIZER,
):
    """
    Compute the CLOCs LiDAR-distance feature:

        sqrt(x^2 + y^2) / normalizer

    The pinned CLOCs implementation uses 82.0.
    """

    lidar_boxes = np.asarray(
        lidar_boxes,
        dtype=np.float32,
    )

    if (
        lidar_boxes.ndim != 2
        or lidar_boxes.shape[1] < 2
    ):
        raise ValueError(
            "lidar_boxes must have shape [N, >=2]"
        )

    if normalizer <= 0:
        raise ValueError(
            "normalizer must be positive"
        )

    xy = lidar_boxes[:, :2]

    distance = np.linalg.norm(
        xy,
        axis=1,
    )

    return (
        distance / float(normalizer)
    ).astype(np.float32)


def build_clocs_feature_rows(
    projected_boxes,
    scores_3d,
    boxes_2d,
    scores_2d,
    distance_82,
):
    """
    Build candidate-level CLOCs feature rows.

    Matched row:
        [projected_2D_IoU,
         3D_detector_score,
         2D_detector_score,
         distance/82]

    Unmatched row:
        [-10,
         3D_detector_score,
         -10,
         distance/82]

    A sentinel row is created only when a 3D candidate
    overlaps no 2D detection.

    This follows the intended CLOCs semantics established
    by the Stage-5 unmatched-candidate audit.
    """

    projected_boxes = np.asarray(
        projected_boxes,
        dtype=np.float32,
    )

    boxes_2d = np.asarray(
        boxes_2d,
        dtype=np.float32,
    )

    scores_3d = np.asarray(
        scores_3d,
        dtype=np.float32,
    ).reshape(-1)

    scores_2d = np.asarray(
        scores_2d,
        dtype=np.float32,
    ).reshape(-1)

    distance_82 = np.asarray(
        distance_82,
        dtype=np.float32,
    ).reshape(-1)

    if (
        projected_boxes.ndim != 2
        or projected_boxes.shape[1] != 4
    ):
        raise ValueError(
            "projected_boxes must have shape [N, 4]"
        )

    if (
        boxes_2d.ndim != 2
        or boxes_2d.shape[1] != 4
    ):
        raise ValueError(
            "boxes_2d must have shape [M, 4]"
        )

    num_3d = projected_boxes.shape[0]
    num_2d = boxes_2d.shape[0]

    if scores_3d.shape[0] != num_3d:
        raise ValueError(
            "scores_3d length must match projected_boxes"
        )

    if distance_82.shape[0] != num_3d:
        raise ValueError(
            "distance_82 length must match projected_boxes"
        )

    if scores_2d.shape[0] != num_2d:
        raise ValueError(
            "scores_2d length must match boxes_2d"
        )

    ious = pairwise_iou_xyxy(
        projected_boxes,
        boxes_2d,
    )

    overlap = ious > 0.0

    pair_indices = np.argwhere(
        overlap
    )

    candidate_positive = (
        pair_indices[:, 0]
        if len(pair_indices)
        else np.empty(
            0,
            dtype=np.int64,
        )
    )

    detector_positive = (
        pair_indices[:, 1]
        if len(pair_indices)
        else np.empty(
            0,
            dtype=np.int64,
        )
    )

    if len(pair_indices):
        positive_features = np.column_stack([
            ious[
                candidate_positive,
                detector_positive,
            ],
            scores_3d[
                candidate_positive
            ],
            scores_2d[
                detector_positive
            ],
            distance_82[
                candidate_positive
            ],
        ]).astype(np.float32)

    else:
        positive_features = np.empty(
            (0, 4),
            dtype=np.float32,
        )

    has_overlap = overlap.any(
        axis=1
    )

    unmatched_candidates = np.flatnonzero(
        ~has_overlap
    )

    sentinel_features = np.column_stack([
        np.full(
            len(unmatched_candidates),
            UNMATCHED_IOU,
            dtype=np.float32,
        ),
        scores_3d[
            unmatched_candidates
        ],
        np.full(
            len(unmatched_candidates),
            UNMATCHED_2D_SCORE,
            dtype=np.float32,
        ),
        distance_82[
            unmatched_candidates
        ],
    ]).astype(np.float32)

    features = np.concatenate(
        [
            positive_features,
            sentinel_features,
        ],
        axis=0,
    )

    candidate_index = np.concatenate(
        [
            candidate_positive,
            unmatched_candidates,
        ]
    ).astype(np.int32)

    detector_2d_index = np.concatenate(
        [
            detector_positive,
            np.full(
                len(unmatched_candidates),
                -1,
                dtype=np.int64,
            ),
        ]
    ).astype(np.int32)

    row_type = np.concatenate(
        [
            np.ones(
                len(positive_features),
                dtype=np.uint8,
            ),
            np.zeros(
                len(sentinel_features),
                dtype=np.uint8,
            ),
        ]
    )

    order = np.lexsort(
        (
            detector_2d_index,
            candidate_index,
        )
    )

    return {
        "features": features[order],
        "candidate_index": candidate_index[order],
        "detector_2d_index": detector_2d_index[order],
        "row_type": row_type[order],
        "pairwise_iou": ious,
    }
