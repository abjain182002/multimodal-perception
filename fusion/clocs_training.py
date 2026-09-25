#!/usr/bin/env python3

import numpy as np


POSITIVE_IOU_THRESHOLD = 0.70
NEGATIVE_IOU_THRESHOLD = 0.50

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
IGNORE_LABEL = -1


def build_candidate_targets(
    best_iou,
    positive_iou_threshold=POSITIVE_IOU_THRESHOLD,
    negative_iou_threshold=NEGATIVE_IOU_THRESHOLD,
):
    """
    Build CLOCs candidate-level classification targets.

    For the KITTI Car fusion protocol:

        IoU >= 0.70  -> positive
        IoU <= 0.50  -> negative
        otherwise    -> ignored

    Parameters
    ----------
    best_iou:
        One maximum 3D IoU value per PointPillars candidate.

    Returns
    -------
    dict containing:
        labels
        targets
        positive_mask
        negative_mask
        ignore_mask
        num_positive
        num_negative
        num_ignored

    labels use:
         1 = positive
         0 = negative
        -1 = ignored

    targets are binary 0/1 values. Ignored candidates have target
    value 0 but must later receive zero classification weight.
    """

    best_iou = np.asarray(
        best_iou,
        dtype=np.float32,
    ).reshape(-1)

    positive_iou_threshold = float(
        positive_iou_threshold
    )

    negative_iou_threshold = float(
        negative_iou_threshold
    )

    if (
        negative_iou_threshold
        >= positive_iou_threshold
    ):
        raise ValueError(
            "negative_iou_threshold must be less than "
            "positive_iou_threshold"
        )

    if not np.isfinite(
        best_iou
    ).all():
        raise ValueError(
            "best_iou contains non-finite values"
        )

    if np.any(
        best_iou < 0.0
    ):
        raise ValueError(
            "best_iou cannot contain negative values"
        )

    if np.any(
        best_iou > 1.0
    ):
        raise ValueError(
            "best_iou cannot exceed 1.0"
        )

    positive_mask = (
        best_iou
        >= positive_iou_threshold
    )

    negative_mask = (
        best_iou
        <= negative_iou_threshold
    )

    ignore_mask = ~(
        positive_mask
        | negative_mask
    )

    labels = np.full(
        best_iou.shape,
        IGNORE_LABEL,
        dtype=np.int8,
    )

    labels[
        negative_mask
    ] = NEGATIVE_LABEL

    labels[
        positive_mask
    ] = POSITIVE_LABEL

    targets = positive_mask.astype(
        np.float32
    )

    return {
        "best_iou": best_iou,
        "labels": labels,
        "targets": targets,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "ignore_mask": ignore_mask,
        "num_positive": int(
            positive_mask.sum()
        ),
        "num_negative": int(
            negative_mask.sum()
        ),
        "num_ignored": int(
            ignore_mask.sum()
        ),
    }


FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=FOCAL_ALPHA,
    gamma=FOCAL_GAMMA,
    weights=None,
    reduction="none",
):
    """
    Binary sigmoid focal classification loss used by CLOCs.

    Parameters
    ----------
    logits:
        Raw candidate-level fusion logits.

    targets:
        Binary targets with the same shape as logits:
            1 = positive
            0 = negative

    alpha:
        Positive-class balancing factor.

    gamma:
        Focusing parameter.

    weights:
        Optional per-candidate classification weights.
        Ignored candidates will later receive weight zero.

    reduction:
        "none", "sum", or "mean".

    Notes
    -----
    This function expects ignored candidates to be handled through
    zero weights rather than a special target value.
    """

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for CLOCs focal loss."
        ) from exc

    if not torch.is_tensor(
        logits
    ):
        logits = torch.as_tensor(
            logits,
            dtype=torch.float32,
        )

    if not torch.is_tensor(
        targets
    ):
        targets = torch.as_tensor(
            targets,
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        targets = targets.to(
            dtype=logits.dtype,
            device=logits.device,
        )

    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have identical shapes"
        )

    if logits.numel() == 0:
        raise ValueError(
            "At least one candidate is required"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise ValueError(
            "logits contain non-finite values"
        )

    if not torch.isfinite(
        targets
    ).all():
        raise ValueError(
            "targets contain non-finite values"
        )

    if torch.any(
        targets < 0.0
    ) or torch.any(
        targets > 1.0
    ):
        raise ValueError(
            "targets must be in the range [0, 1]"
        )

    alpha = float(
        alpha
    )

    gamma = float(
        gamma
    )

    if (
        alpha < 0.0
        or alpha > 1.0
    ):
        raise ValueError(
            "alpha must be in the range [0, 1]"
        )

    if gamma < 0.0:
        raise ValueError(
            "gamma must be non-negative"
        )

    probability = torch.sigmoid(
        logits
    )

    # For positives:
    #   p_t error term = 1 - p
    #
    # For negatives:
    #   p_t error term = p

    modulation = (
        targets
        * (1.0 - probability)
        + (1.0 - targets)
        * probability
    )

    alpha_factor = (
        targets
        * alpha
        + (1.0 - targets)
        * (1.0 - alpha)
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    loss = (
        alpha_factor
        * modulation.pow(
            gamma
        )
        * bce
    )

    if weights is not None:
        if not torch.is_tensor(
            weights
        ):
            weights = torch.as_tensor(
                weights,
                dtype=logits.dtype,
                device=logits.device,
            )
        else:
            weights = weights.to(
                dtype=logits.dtype,
                device=logits.device,
            )

        if weights.shape != logits.shape:
            raise ValueError(
                "weights must have the same shape as logits"
            )

        if not torch.isfinite(
            weights
        ).all():
            raise ValueError(
                "weights contain non-finite values"
            )

        if torch.any(
            weights < 0.0
        ):
            raise ValueError(
                "weights cannot be negative"
            )

        loss = (
            loss
            * weights
        )

    if reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        return loss.mean()

    raise ValueError(
        "reduction must be 'none', 'sum', or 'mean'"
    )


def build_classification_weights(
    labels,
    positive_weight=1.0,
    negative_weight=1.0,
):
    """
    Build CLOCs candidate classification weights.

    Semantics:
         1 = positive
         0 = negative
        -1 = ignored

    Positive and negative weights are normalized by the number
    of positive candidates, clamped to a minimum of one.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for CLOCs classification weights."
        ) from exc

    if not torch.is_tensor(
        labels
    ):
        labels = torch.as_tensor(
            labels,
            dtype=torch.int64,
        )
    else:
        labels = labels.to(
            dtype=torch.int64,
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape [N]"
        )

    if labels.numel() == 0:
        raise ValueError(
            "At least one candidate label is required"
        )

    valid_labels = (
        (labels == IGNORE_LABEL)
        | (labels == NEGATIVE_LABEL)
        | (labels == POSITIVE_LABEL)
    )

    if not torch.all(
        valid_labels
    ):
        raise ValueError(
            "labels must contain only -1, 0, or 1"
        )

    positive_weight = float(
        positive_weight
    )

    negative_weight = float(
        negative_weight
    )

    if (
        positive_weight < 0.0
        or negative_weight < 0.0
    ):
        raise ValueError(
            "classification weights cannot be negative"
        )

    positive_mask = (
        labels == POSITIVE_LABEL
    )

    negative_mask = (
        labels == NEGATIVE_LABEL
    )

    ignore_mask = (
        labels == IGNORE_LABEL
    )

    dtype = torch.float32

    weights = (
        positive_mask.to(dtype)
        * positive_weight
        + negative_mask.to(dtype)
        * negative_weight
    )

    num_positive = positive_mask.sum()

    normalizer = torch.clamp(
        num_positive.to(dtype),
        min=1.0,
    )

    weights = (
        weights
        / normalizer
    )

    return {
        "weights": weights,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "ignore_mask": ignore_mask,
        "num_positive": int(
            positive_mask.sum().item()
        ),
        "num_negative": int(
            negative_mask.sum().item()
        ),
        "num_ignored": int(
            ignore_mask.sum().item()
        ),
        "normalizer": float(
            normalizer.item()
        ),
    }
