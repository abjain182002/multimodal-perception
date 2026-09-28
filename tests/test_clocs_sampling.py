import unittest
import numpy as np
from fusion.clocs_sampling import PILOT_IDS, split_frames, select_candidates


class SamplingTests(unittest.TestCase):
    def test_ratio_positives_ignore_hard_and_repeatability(self):
        labels = np.array([1, 1, -1] + [0] * 100)
        scores = np.arange(103, dtype=float)
        chosen = select_candidates(labels, scores, 8, 7, 1, '000001')
        self.assertEqual(len(chosen), 18)
        self.assertTrue({0, 1}.issubset(chosen))
        self.assertNotIn(2, chosen)
        self.assertTrue(set(range(95, 103)).issubset(chosen))
        self.assertEqual(len(set(chosen)), len(chosen))
        np.testing.assert_array_equal(chosen, select_candidates(labels, scores, 8, 7, 1, '000001'))
        self.assertFalse(np.array_equal(chosen, select_candidates(labels, scores, 8, 7, 2, '000001')))

    def test_background_and_small_negative_pool(self):
        self.assertEqual(len(select_candidates(np.zeros(1000), np.arange(1000), 8, 1, 1, 'a')), 256)
        np.testing.assert_array_equal(select_candidates([1, 0, -1], [0.5, 0.4, 0.2], 8, 1, 1, 'a'), [0, 1])

    def test_baseline_keeps_all_valid(self):
        np.testing.assert_array_equal(select_candidates([1, 0, -1, 0], [0.1]*4, 0, 1, 1, 'a'), [0, 1, 3])

    def test_split_excludes_old_pilot_from_dev(self):
        ids = sorted(PILOT_IDS | {f'{i:06}' for i in range(8000, 8112)})
        train, dev = split_frames(ids)
        self.assertEqual((len(train), len(dev)), (96, 32))
        self.assertFalse(set(train) & set(dev))
        self.assertFalse(PILOT_IDS & set(dev))
        self.assertEqual(set(train) | set(dev), set(ids))
        self.assertEqual((train, dev), split_frames(ids[::-1]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_candidates([2], [0.5], 8, 1, 1, 'a')
        with self.assertRaises(ValueError):
            select_candidates([0], [float('nan')], 8, 1, 1, 'a')
        with self.assertRaises(ValueError):
            split_frames(['000000'] * 128)
