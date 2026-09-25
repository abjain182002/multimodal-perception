#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from fusion.clocs_training import (
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    build_classification_weights,
    sigmoid_focal_loss,
)


def numpy_focal_loss_reference(
    logits,
    targets,
    weights=None,
    alpha=0.25,
    gamma=2.0,
):
    """
    Independent NumPy implementation of binary sigmoid focal loss.

    This intentionally does not call the project focal-loss function
    or PyTorch binary-cross-entropy helpers.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    probability = (
        1.0
        / (
            1.0
            + np.exp(
                -logits
            )
        )
    )

    p_t = (
        targets
        * probability
        + (1.0 - targets)
        * (1.0 - probability)
    )

    alpha_t = (
        targets
        * alpha
        + (1.0 - targets)
        * (1.0 - alpha)
    )

    loss = (
        -alpha_t
        * np.power(
            1.0 - p_t,
            gamma,
        )
        * np.log(
            p_t
        )
    )

    if weights is not None:
        loss = (
            loss
            * np.asarray(
                weights,
                dtype=np.float64,
            )
        )

    return loss


class TestCLOCsLossReference(unittest.TestCase):

    def test_elementwise_loss_matches_numpy_reference(self):
        logits_np = np.array(
            [
                -2.3,
                -0.7,
                0.0,
                0.8,
                2.4,
            ],
            dtype=np.float64,
        )

        targets_np = np.array(
            [
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
            ],
            dtype=np.float64,
        )

        expected = numpy_focal_loss_reference(
            logits=logits_np,
            targets=targets_np,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
        )

        actual = sigmoid_focal_loss(
            logits=torch.tensor(
                logits_np,
                dtype=torch.float64,
            ),
            targets=torch.tensor(
                targets_np,
                dtype=torch.float64,
            ),
            reduction="none",
        )

        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_weighted_normalized_loss_matches_numpy_reference(self):
        labels = np.array(
            [
                1,
                0,
                -1,
                1,
                0,
            ],
            dtype=np.int64,
        )

        targets_np = np.array(
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )

        logits_np = np.array(
            [
                1.4,
                -0.4,
                0.9,
                -0.2,
                0.3,
            ],
            dtype=np.float64,
        )

        weight_result = build_classification_weights(
            labels
        )

        weights_np = (
            weight_result["weights"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        expected_elementwise = (
            numpy_focal_loss_reference(
                logits=logits_np,
                targets=targets_np,
                weights=weights_np,
                alpha=FOCAL_ALPHA,
                gamma=FOCAL_GAMMA,
            )
        )

        actual_elementwise = sigmoid_focal_loss(
            logits=torch.tensor(
                logits_np,
                dtype=torch.float64,
            ),
            targets=torch.tensor(
                targets_np,
                dtype=torch.float64,
            ),
            weights=weight_result["weights"],
            reduction="none",
        )

        np.testing.assert_allclose(
            actual_elementwise
            .detach()
            .cpu()
            .numpy(),
            expected_elementwise,
            rtol=1e-12,
            atol=1e-12,
        )

        expected_sum = float(
            expected_elementwise.sum()
        )

        actual_sum = sigmoid_focal_loss(
            logits=torch.tensor(
                logits_np,
                dtype=torch.float64,
            ),
            targets=torch.tensor(
                targets_np,
                dtype=torch.float64,
            ),
            weights=weight_result["weights"],
            reduction="sum",
        )

        self.assertAlmostEqual(
            actual_sum.item(),
            expected_sum,
            places=12,
        )

        # The ignored candidate is index 2.
        self.assertEqual(
            weights_np[2],
            0.0,
        )

        self.assertEqual(
            expected_elementwise[2],
            0.0,
        )

        self.assertEqual(
            weight_result["normalizer"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
