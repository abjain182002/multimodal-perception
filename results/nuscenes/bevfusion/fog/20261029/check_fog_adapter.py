"""Meaningful CPU checks for adapter state and GPU input preparation helpers.

Requires the private, verified pilot directory; never requires a GPU.
"""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fog_transforms import FogAdapter, NumericTableUnpickler, seed
import fog_gpu_smoke as smoke

PILOT = None


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.adapter = FogAdapter(PILOT / 'upstream')

    def test_global_camera_rng_and_identity(self):
        rgb = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        before = np.random.get_state()
        y = self.adapter.camera(rgb, 2, 'samples/CAM_FRONT/test.jpg')
        smoke.same(before, np.random.get_state())
        np.testing.assert_array_equal(y, self.adapter.camera(rgb, 2, 'samples/CAM_FRONT/test.jpg', reference=True))
        np.testing.assert_array_equal(rgb, self.adapter.camera(rgb, 0, 'samples/CAM_FRONT/test.jpg'))
        self.assertNotEqual(seed('camera', 'path1', 1), seed('camera', 'path2', 1))
        self.assertNotEqual(seed('camera', 'path1', 1), seed('camera', 'path1', 2))

    def test_raw_feature_types_and_numeric_only_table_loader(self):
        with self.assertRaises(AssertionError):
            self.adapter.points(np.zeros((2, 5), np.float64), 1, 'path')
        with self.assertRaises(AssertionError):
            self.adapter.points(np.zeros((2, 4), np.float32), 1, 'path')
        with self.assertRaises(ValueError):
            NumericTableUnpickler(io.BytesIO(pickle.dumps(1 + 2j))).load()

    def test_unambiguous_annotation_resolution(self):
        allowed = {'samples/LIDAR_TOP/same.bin', 'sweeps/LIDAR_TOP/same.bin'}
        self.assertEqual(smoke.relative_path('/old/root/sweeps/LIDAR_TOP/same.bin', allowed), 'sweeps/LIDAR_TOP/same.bin')
        with self.assertRaises(AssertionError): smoke.relative_path('same.bin', allowed)
        with self.assertRaises(AssertionError): smoke.relative_path('/wrong/unknown.bin', allowed)

    def test_pilot_subset_and_non_path_geometry_preservation(self):
        selected = json.loads((PILOT / 'pilot_selection.json').read_text())['selected']
        tokens = (PILOT / 'frozen_population/sample_tokens.txt').read_text().splitlines()
        row = {'token': tokens[0], 'timestamp': 123456,
            'lidar_points': {'lidar_path': Path(selected['lidar_files'][0]).name},
            'images': {Path(p).parts[1]: {'img_path': '/old/dataset/' + p, 'cam2img': np.eye(3, dtype=np.float32)}
                       for p in selected['camera_files']},
            'lidar_sweeps': [{'lidar_points': {'lidar_path': '/old/dataset/' + p}, 'lidar2sensor': np.eye(4)}
                             for p in selected['lidar_files'][1:]],
            'ego2global': np.eye(4), 'instances': [{'bbox_3d': [1., 2., 3., 4., 5., 6., 0.]}]}
        rows = [dict(copy.deepcopy(row), token=t) for t in tokens]
        obj = {'metainfo': {'classes': ('car', 'pedestrian'), 'dataset': 'synthetic-test'}, 'data_list': rows}
        data = pickle.dumps(obj, protocol=4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); annotation = d / 'source.pkl'; annotation.write_bytes(data)
            with patch.object(smoke, 'ANNOTATION_SHA', hashlib.sha256(data).hexdigest()):
                smoke.prepare(SimpleNamespace(pilot=PILOT, annotation=annotation, prepared=d / 'prepared'))
            audit = json.loads((d / 'prepared/pilot_input_audit.json').read_text())
            self.assertEqual(len(audit['conditions']), 4)
            for condition in audit['conditions']:
                actual = pickle.loads(Path(condition['annotation']).read_bytes())
                self.assertEqual(len(actual['data_list']), 1)
                self.assertEqual(actual['data_list'][0]['token'], tokens[0])
                np.testing.assert_array_equal(actual['data_list'][0]['ego2global'], row['ego2global'])
                self.assertEqual(len(condition['source_files']), 16)
                self.assertTrue(condition['non_path_row_fields_unchanged'])
                for file in condition['source_files'].values():
                    self.assertEqual(hashlib.sha256(Path(file['path']).read_bytes()).hexdigest(), file['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', type=Path, required=True)
    args, remaining = parser.parse_known_args(); PILOT = args.pilot
    unittest.main(argv=['check_fog_adapter', *remaining])
