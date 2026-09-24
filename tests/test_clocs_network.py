#!/usr/bin/env python3

import unittest

import torch
from torch import nn

from fusion.clocs_network import (
    CLOCsFusionNet,
    candidate_wise_max,
)


class TestCLOCsFusionNetwork(unittest.TestCase):

    def test_architecture_matches_clocs(self):
        model = CLOCsFusionNet()

        conv_layers = [
            layer
            for layer in model.fuse_2d_3d
            if isinstance(layer, nn.Conv2d)
        ]

        relu_layers = [
            layer
            for layer in model.fuse_2d_3d
            if isinstance(layer, nn.ReLU)
        ]

        self.assertEqual(
            len(conv_layers),
            4,
        )

        self.assertEqual(
            len(relu_layers),
            3,
        )

        expected_channels = [
            (4, 18),
            (18, 36),
            (36, 36),
            (36, 1),
        ]

        actual_channels = [
            (
                layer.in_channels,
                layer.out_channels,
            )
            for layer in conv_layers
        ]

        self.assertEqual(
            actual_channels,
            expected_channels,
        )

        for layer in conv_layers:
            self.assertEqual(
                layer.kernel_size,
                (1, 1),
            )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        self.assertEqual(
            parameter_count,
            2143,
        )

    def test_candidate_max_matches_legacy_dense_pooling(self):
        """
        Compare the dynamic aggregation against the equivalent
        legacy CLOCs detector-by-candidate dense tensor.
        """

        row_logits = torch.tensor(
            [
                0.20,
                0.80,
                -0.30,
                0.40,
                0.10,
                1.20,
            ],
            dtype=torch.float32,
        )

        candidate_index = torch.tensor(
            [
                0,
                0,
                1,
                1,
                1,
                2,
            ],
            dtype=torch.long,
        )

        detector_index = torch.tensor(
            [
                0,
                1,
                0,
                2,
                1,
                0,
            ],
            dtype=torch.long,
        )

        dynamic = candidate_wise_max(
            row_logits=row_logits,
            candidate_index=candidate_index,
            num_candidates=3,
        )

        # Legacy-style dense representation:
        #
        # detector dimension × 3D candidate dimension
        #
        # Missing associations are effectively -infinity
        # before max pooling.

        legacy_dense = torch.full(
            (3, 3),
            -torch.inf,
            dtype=torch.float32,
        )

        legacy_dense[
            detector_index,
            candidate_index,
        ] = row_logits

        legacy = torch.max(
            legacy_dense,
            dim=0,
        ).values

        torch.testing.assert_close(
            dynamic,
            legacy,
            rtol=0.0,
            atol=0.0,
        )

        torch.testing.assert_close(
            dynamic,
            torch.tensor([
                0.80,
                0.40,
                1.20,
            ]),
            rtol=0.0,
            atol=0.0,
        )

    def test_candidate_max_gradient_flows_through_winner(self):
        row_logits = torch.tensor(
            [
                0.20,
                0.80,
                -0.50,
                0.10,
            ],
            dtype=torch.float32,
            requires_grad=True,
        )

        candidate_index = torch.tensor(
            [
                0,
                0,
                1,
                1,
            ],
            dtype=torch.long,
        )

        candidate_logits = candidate_wise_max(
            row_logits=row_logits,
            candidate_index=candidate_index,
            num_candidates=2,
        )

        candidate_logits.sum().backward()

        expected_gradient = torch.tensor(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            row_logits.grad,
            expected_gradient,
            rtol=0.0,
            atol=0.0,
        )

    def test_missing_candidate_raises(self):
        row_logits = torch.tensor(
            [
                0.2,
                0.8,
            ],
            dtype=torch.float32,
        )

        candidate_index = torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.long,
        )

        with self.assertRaises(
            ValueError
        ):
            candidate_wise_max(
                row_logits=row_logits,
                candidate_index=candidate_index,
                num_candidates=3,
            )

    def test_forward_returns_correct_candidate_max(self):
        torch.manual_seed(
            123
        )

        model = CLOCsFusionNet()

        features = torch.tensor(
            [
                [0.8, 0.1, 0.9, 0.2],
                [0.5, 0.2, 0.7, 0.3],
                [0.2, 0.8, 0.4, 0.5],
                [0.4, 0.3, 0.6, 0.7],
                [0.7, 0.4, 0.8, 0.6],
            ],
            dtype=torch.float32,
        )

        candidate_index = torch.tensor(
            [
                0,
                0,
                1,
                2,
                2,
            ],
            dtype=torch.long,
        )

        output = model(
            features=features,
            candidate_index=candidate_index,
            num_candidates=3,
        )

        self.assertEqual(
            output["row_logits"].shape,
            (5,),
        )

        self.assertEqual(
            output["candidate_logits"].shape,
            (3,),
        )

        expected = torch.stack([
            torch.max(
                output["row_logits"][
                    torch.tensor([0, 1])
                ]
            ),
            output["row_logits"][2],
            torch.max(
                output["row_logits"][
                    torch.tensor([3, 4])
                ]
            ),
        ])

        torch.testing.assert_close(
            output["candidate_logits"],
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_invalid_feature_width_raises(self):
        model = CLOCsFusionNet()

        bad_features = torch.zeros(
            (5, 3),
            dtype=torch.float32,
        )

        with self.assertRaises(
            ValueError
        ):
            model.compute_row_logits(
                bad_features
            )


if __name__ == "__main__":
    unittest.main()
