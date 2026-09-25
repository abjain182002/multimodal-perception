import unittest

import numpy as np

from fusion.clocs_iou import best_camera_3d_iou
from fusion.clocs_training import build_candidate_targets


class TestCLOCS3DIoU(unittest.TestCase):
    def setUp(self):
        self.box = np.array([0, 1.5, 20, 4, 1.5, 2, 0], dtype=np.float32)

    def test_exact_match_and_disjoint(self):
        far = self.box.copy()
        far[0] = 20
        result = best_camera_3d_iou(np.stack([self.box, far]),
                                    self.box[None, :])
        np.testing.assert_allclose(result, [1, 0], atol=1e-6)
        np.testing.assert_array_equal(build_candidate_targets(result)["labels"],
                                      [1, 0])

    def test_partial_horizontal_overlap(self):
        shifted = self.box.copy()
        shifted[0] += 2
        # Intersection is 2 x 2 x 1.5; union is 12 + 12 - 6.
        self.assertAlmostEqual(best_camera_3d_iou(
            shifted[None, :], self.box[None, :])[0], 1 / 3, places=6)

    def test_camera_bottom_and_height(self):
        shifted = self.box.copy()
        shifted[1] += 0.75
        # Overlap height is 0.75 of two boxes each 1.5 high.
        self.assertAlmostEqual(best_camera_3d_iou(
            shifted[None, :], self.box[None, :])[0], 1 / 3, places=6)
        shifted[1] += 1.5
        self.assertEqual(best_camera_3d_iou(
            shifted[None, :], self.box[None, :])[0], 0)

    def test_yaw_rotation_and_symmetry(self):
        rotated = self.box.copy()
        rotated[6] = np.pi / 2
        # The 4x2 and 2x4 footprints intersect in a 2x2 square.
        expected = 4 / (8 + 8 - 4)
        self.assertAlmostEqual(best_camera_3d_iou(
            rotated[None, :], self.box[None, :])[0], expected, places=6)
        self.assertAlmostEqual(best_camera_3d_iou(
            self.box[None, :], rotated[None, :])[0], expected, places=6)
        self.assertAlmostEqual(best_camera_3d_iou(
            rotated[None, :], rotated[None, :])[0], 1, places=6)

    def test_asymmetric_rotated_boxes_are_symmetric(self):
        a = self.box.copy()
        b = self.box.copy()
        a[6] = 0.4
        b[6] = -0.8
        b[0] += 1.2
        b[2] += 0.3
        forward = best_camera_3d_iou(a[None, :], b[None, :])[0]
        reverse = best_camera_3d_iou(b[None, :], a[None, :])[0]
        self.assertGreater(forward, 0)
        self.assertLess(forward, 1)
        self.assertAlmostEqual(forward, reverse, places=6)

    def test_best_of_multiple_gt_and_empty_gt(self):
        near = self.box.copy()
        near[0] += 1
        distant = self.box.copy()
        distant[2] += 40
        self.assertAlmostEqual(best_camera_3d_iou(
            near[None, :], np.stack([distant, self.box]))[0], 0.6, places=6)
        np.testing.assert_array_equal(best_camera_3d_iou(
            np.stack([near, distant]), np.empty((0, 7))), [0, 0])
        self.assertEqual(best_camera_3d_iou(
            np.empty((0, 7)), self.box[None, :]).shape, (0,))

    def test_reject_bad_boxes(self):
        for bad in [self.box[:6][None, :],
                    np.full((1, 7), np.nan),
                    np.array([[0, 0, 0, 4, -1, 2, 0]])]:
            with self.assertRaises(ValueError):
                best_camera_3d_iou(bad, self.box[None, :])


if __name__ == "__main__":
    unittest.main()
