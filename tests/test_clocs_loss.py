#!/usr/bin/env python3

import unittest

import torch

from fusion.clocs_training import (
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    build_classification_weights,
    sigmoid_focal_loss,
)


class TestCLOCsLoss(unittest.TestCase):

    def test_default_focal_parameters(self):
        self.assertEqual(
            FOCAL_ALPHA,
            0.25,
        )

        self.assertEqual(
            FOCAL_GAMMA,
            2.0,
        )

    def test_easy_positive_has_smaller_loss(self):
        logits = torch.tensor(
            [
                0.0,
                4.0,
            ],
            dtype=torch.float32,
        )

        targets = torch.ones(
            2,
            dtype=torch.float32,
        )

        loss = sigmoid_focal_loss(
            logits,
            targets,
        )

        self.assertLess(
            loss[1].item(),
            loss[0].item(),
        )

    def test_easy_negative_has_smaller_loss(self):
        logits = torch.tensor(
            [
                0.0,
                -4.0,
            ],
            dtype=torch.float32,
        )

        targets = torch.zeros(
            2,
            dtype=torch.float32,
        )

        loss = sigmoid_focal_loss(
            logits,
            targets,
        )

        self.assertLess(
            loss[1].item(),
            loss[0].item(),
        )

    def test_zero_weight_removes_candidate_loss(self):
        logits = torch.tensor(
            [
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor(
            [
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        )

        weights = torch.tensor(
            [
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        )

        loss = sigmoid_focal_loss(
            logits,
            targets,
            weights=weights,
        )

        self.assertGreater(
            loss[0].item(),
            0.0,
        )

        self.assertEqual(
            loss[1].item(),
            0.0,
        )

    def test_reductions_are_consistent(self):
        logits = torch.tensor(
            [
                -1.0,
                0.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        targets = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        elementwise = sigmoid_focal_loss(
            logits,
            targets,
            reduction="none",
        )

        summed = sigmoid_focal_loss(
            logits,
            targets,
            reduction="sum",
        )

        meaned = sigmoid_focal_loss(
            logits,
            targets,
            reduction="mean",
        )

        torch.testing.assert_close(
            summed,
            elementwise.sum(),
        )

        torch.testing.assert_close(
            meaned,
            elementwise.mean(),
        )

    def test_focal_loss_backpropagation(self):
        logits = torch.tensor(
            [
                -1.0,
                0.5,
                1.2,
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        targets = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        loss = sigmoid_focal_loss(
            logits,
            targets,
            reduction="sum",
        )

        loss.backward()

        self.assertTrue(
            torch.isfinite(
                logits.grad
            ).all()
        )

        self.assertGreater(
            torch.count_nonzero(
                logits.grad
            ).item(),
            0,
        )

    def test_positive_count_normalization(self):
        result = build_classification_weights(
            [
                1,
                0,
                -1,
                1,
            ]
        )

        torch.testing.assert_close(
            result["weights"],
            torch.tensor(
                [
                    0.5,
                    0.5,
                    0.0,
                    0.5,
                ],
                dtype=torch.float32,
            ),
            rtol=0.0,
            atol=0.0,
        )

        self.assertEqual(
            result["num_positive"],
            2,
        )

        self.assertEqual(
            result["num_negative"],
            1,
        )

        self.assertEqual(
            result["num_ignored"],
            1,
        )

        self.assertEqual(
            result["normalizer"],
            2.0,
        )

    def test_no_positive_uses_unit_normalizer(self):
        result = build_classification_weights(
            [
                0,
                0,
                -1,
            ]
        )

        torch.testing.assert_close(
            result["weights"],
            torch.tensor(
                [
                    1.0,
                    1.0,
                    0.0,
                ],
                dtype=torch.float32,
            ),
            rtol=0.0,
            atol=0.0,
        )

        self.assertEqual(
            result["normalizer"],
            1.0,
        )

    def test_invalid_labels_raise(self):
        with self.assertRaises(
            ValueError
        ):
            build_classification_weights(
                [
                    0,
                    1,
                    2,
                ]
            )


if __name__ == "__main__":
    unittest.main()
