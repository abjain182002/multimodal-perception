import unittest

import numpy as np

from scripts.audit_clocs_iou_reference import compare_iou, representative_indices


class ReferenceIoUAuditTests(unittest.TestCase):
    def test_sampling_covers_positive_ignore_overlap_negative_and_zero(self):
        values = np.array([0, 0, .2, .4, .49, .51, .6, .69, .7, .8, .9],
                          dtype=np.float32)
        indices = representative_indices(values, per_group=2)
        self.assertTrue(np.isin([0, 2, 5, 8, 10], indices).all())
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertIn(4, indices)  # nearest negative to 0.50
        self.assertIn(8, indices)  # boundary positive at 0.70

    def test_compare_checks_3d_iou_and_label_parity(self):
        ours = np.array([0, .42, .61, .78])
        indices = np.array([0, 5, 12, 31])
        good = compare_iou(ours, [0, .42001, .61, .78001], indices)
        self.assertEqual(good['status'], 'PASS')
        self.assertEqual(good['sample_size'], 4)
        bad = compare_iou(ours, [0, .62, .61, .78001], indices)
        self.assertEqual(bad['status'], 'FAIL')
        self.assertEqual(bad['label_disagreements'], 1)
        self.assertEqual(bad['worst_pairs'][0]['candidate_index'], 5)

    def test_shape_and_reference_validation(self):
        with self.assertRaises(ValueError):
            compare_iou([.1], [.1, .2], [1])
        with self.assertRaises(ValueError):
            compare_iou([.1], [float('nan')], [1])
        with self.assertRaises(ValueError):
            representative_indices([0, .2], per_group=0)


if __name__ == '__main__':
    unittest.main()
