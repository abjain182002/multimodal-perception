#!/usr/bin/env python3

import torch
from torch import nn


FEATURE_DIM = 4
HIDDEN_CHANNELS = (
    18,
    36,
    36,
)


def candidate_wise_max(
    row_logits,
    candidate_index,
    num_candidates,
):
    """
    Aggregate CLOCs association logits with a maximum for each
    3D candidate.

    This is the dynamic equivalent of scattering association
    logits into the legacy detector-by-candidate tensor and
    max-pooling over the 2D-detection dimension.
    """

    if row_logits.ndim != 1:
        raise ValueError(
            "row_logits must have shape [R]"
        )

    candidate_index = torch.as_tensor(
        candidate_index,
        device=row_logits.device,
    )

    if candidate_index.ndim != 1:
        raise ValueError(
            "candidate_index must have shape [R]"
        )

    if candidate_index.shape[0] != row_logits.shape[0]:
        raise ValueError(
            "candidate_index length must match row_logits"
        )

    num_candidates = int(
        num_candidates
    )

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive"
        )

    if candidate_index.numel() == 0:
        raise ValueError(
            "At least one association row is required"
        )

    candidate_index = candidate_index.to(
        dtype=torch.long,
    )

    if torch.any(
        candidate_index < 0
    ):
        raise ValueError(
            "candidate_index cannot contain negative values"
        )

    if torch.any(
        candidate_index >= num_candidates
    ):
        raise ValueError(
            "candidate_index exceeds num_candidates"
        )

    candidate_logits = torch.full(
        (num_candidates,),
        -torch.inf,
        dtype=row_logits.dtype,
        device=row_logits.device,
    )

    candidate_logits.scatter_reduce_(
        dim=0,
        index=candidate_index,
        src=row_logits,
        reduce="amax",
        include_self=True,
    )

    missing = ~torch.isfinite(
        candidate_logits
    )

    if torch.any(missing):
        missing_indices = (
            torch.nonzero(
                missing,
                as_tuple=False,
            )
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Every 3D candidate must have at least one "
            f"CLOCs feature row. Missing: {missing_indices[:10]}"
        )

    return candidate_logits


class CLOCsFusionNet(nn.Module):
    """
    CLOCs candidate-level fusion network.

    Input feature rows contain:

        [projected_2d_iou,
         3d_detector_score,
         2d_detector_score,
         normalized_lidar_distance]

    Each association row is processed independently by the
    original 1x1-convolution architecture. Association logits
    belonging to the same 3D candidate are then max-aggregated.
    """

    def __init__(self):
        super().__init__()

        self.fuse_2d_3d = nn.Sequential(
            nn.Conv2d(
                FEATURE_DIM,
                HIDDEN_CHANNELS[0],
                kernel_size=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                HIDDEN_CHANNELS[0],
                HIDDEN_CHANNELS[1],
                kernel_size=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                HIDDEN_CHANNELS[1],
                HIDDEN_CHANNELS[2],
                kernel_size=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                HIDDEN_CHANNELS[2],
                1,
                kernel_size=1,
            ),
        )

    def compute_row_logits(
        self,
        features,
    ):
        """
        Compute one learned fusion logit per association row.
        """

        if features.ndim != 2:
            raise ValueError(
                "features must have shape [R, 4]"
            )

        if features.shape[1] != FEATURE_DIM:
            raise ValueError(
                "features must contain exactly four CLOCs features"
            )

        if features.shape[0] == 0:
            raise ValueError(
                "At least one CLOCs feature row is required"
            )

        # [R, 4]
        #     →
        # [1, 4, 1, R]
        #
        # This preserves the original CLOCs 1x1 Conv2d
        # formulation while allowing a dynamic number of rows.

        network_input = (
            features
            .transpose(0, 1)
            .unsqueeze(0)
            .unsqueeze(2)
        )

        logits = self.fuse_2d_3d(
            network_input
        )

        # [1, 1, 1, R] → [R]

        return logits[
            0,
            0,
            0,
            :,
        ]

    def forward(
        self,
        features,
        candidate_index,
        num_candidates,
    ):
        """
        Return both association-level and candidate-level logits.
        """

        row_logits = self.compute_row_logits(
            features
        )

        candidate_logits = candidate_wise_max(
            row_logits=row_logits,
            candidate_index=candidate_index,
            num_candidates=num_candidates,
        )

        return {
            "row_logits": row_logits,
            "candidate_logits": candidate_logits,
        }
