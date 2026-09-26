import unittest

import numpy as np

from fusion.clocs_iou import best_camera_3d_iou
from scripts.audit_clocs_iou_scipy import scipy_best_camera_3d_iou


class SciPyGeometryParityTests(unittest.TestCase):
    def test_rotated_height_and_nonoverlap_cases(self):
        gt = np.array([[0, 1.5, 20, 4, 1.5, 2, 0]], dtype=np.float64)
        candidates = np.repeat(gt, 7, axis=0)
        candidates[1, 0] += 2
        candidates[2, 1] += .75
        candidates[3, 6] = np.pi / 2
        candidates[4, 0] += 20
        candidates[5, 1] += 1.5
        candidates[6, 6] = .4
        expected = [1, 1/3, 1/3, 1/3, 0, 0,
                    best_camera_3d_iou(candidates[6:7], gt)[0]]
        actual = scipy_best_camera_3d_iou(candidates, gt)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_asymmetric_rotations_and_offset_multi_gt(self):
        gt = np.array([[0, 1.5, 20, 4, 1.5, 2, .75],
                       [8, 2.2, 32, 5.2, 2.1, 2.4, -.6]], dtype=np.float64)
        boxes = np.array([[.9, 1.7, 20.2, 3.8, 1.7, 1.8, -.4],
                          [8.2, 2.0, 32.4, 5, 1.8, 2.6, -.5],
                          [50, 1.5, 20, 4, 1.5, 2, .75]], dtype=np.float64)
        ours = best_camera_3d_iou(boxes, gt)
        independently = scipy_best_camera_3d_iou(boxes, gt)
        np.testing.assert_allclose(ours, independently, rtol=0, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
