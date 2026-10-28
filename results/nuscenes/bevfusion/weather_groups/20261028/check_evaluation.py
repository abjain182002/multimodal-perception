"""Checks against SDK 1.2.0 algorithms and synthetic nuScenes metadata."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

HERE = Path(__file__).parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


e = module('weather_eval', 'evaluate_weather_groups.py')
r = module('weather_run', 'run_weather_groups_colab.py')


def tables(root):
    """Include a temporal GT link to a frame outside the selected scene."""
    from PIL import Image
    source = {
        'category': [{'token': 'cat', 'name': 'vehicle.car', 'description': 'car'}],
        'attribute': [{'token': 'attr', 'name': 'vehicle.parked', 'description': 'parked'}],
        'visibility': [{'token': '4', 'level': 'v80-100', 'description': 'visible'}],
        'instance': [{'token': 'inst', 'category_token': 'cat', 'nbr_annotations': 3,
                      'first_annotation_token': 'aa', 'last_annotation_token': 'ac'}],
        'sensor': [{'token': 'lidar', 'modality': 'lidar', 'channel': 'LIDAR_TOP'}],
        'calibrated_sensor': [{'token': 'cs', 'sensor_token': 'lidar',
            'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0], 'camera_intrinsic': []}],
        'ego_pose': [{'token': 'e' + t, 'translation': [0, 0, 0],
                      'rotation': [1, 0, 0, 0], 'timestamp': i * 500000} for i, t in enumerate('abc')],
        'log': [{'token': 'log', 'logfile': 'synthetic', 'vehicle': 'test',
                 'date_captured': '2026-10-28', 'location': 'boston-seaport'}],
        'scene': [{'token': 'val', 'name': 'scene-0003', 'log_token': 'log', 'nbr_samples': 2,
                   'first_sample_token': 'a', 'last_sample_token': 'b', 'description': 'Night'},
                  {'token': 'train', 'name': 'scene-0001', 'log_token': 'log', 'nbr_samples': 1,
                   'first_sample_token': 'c', 'last_sample_token': 'c', 'description': 'Road'}],
        'sample': [{'token': t, 'timestamp': i * 500000,
                    'scene_token': 'val' if t in 'ab' else 'train',
                    'prev': '' if t != 'b' else 'a', 'next': 'b' if t == 'a' else ''}
                   for i, t in enumerate('abc')],
        'sample_data': [{'token': 'sd' + t, 'sample_token': t, 'ego_pose_token': 'e' + t,
            'calibrated_sensor_token': 'cs', 'timestamp': i * 500000, 'is_key_frame': True,
            'filename': f'samples/LIDAR_TOP/{t}.pcd.bin', 'fileformat': 'pcd',
            'width': 0, 'height': 0, 'prev': '', 'next': ''} for i, t in enumerate('abc')],
        'sample_annotation': [{'token': 'a' + t, 'sample_token': t, 'instance_token': 'inst',
            'attribute_tokens': ['attr'], 'visibility_token': '4', 'translation': [10 + i, 0, 0],
            'size': [2, 4, 2], 'rotation': [1, 0, 0, 0], 'num_lidar_pts': 10, 'num_radar_pts': 0,
            'prev': '' if i == 0 else 'a' + 'abc'[i - 1],
            'next': '' if i == 2 else 'a' + 'abc'[i + 1]} for i, t in enumerate('abc')],
        'map': [{'token': 'map', 'log_tokens': ['log'], 'category': 'semantic_prior',
                 'filename': 'maps/test.png'}]}
    (root / 'v1.0-trainval').mkdir(parents=True)
    for name, rows in source.items():
        (root / 'v1.0-trainval' / (name + '.json')).write_text(json.dumps(rows))
    (root / 'maps').mkdir()
    Image.new('L', (4, 4), 0).save(root / 'maps/test.png')
    return {str(f.relative_to(root)): {'bytes': f.stat().st_size, 'sha256': r.file_sha(f)}
            for f in root.rglob('*') if f.is_file()}


class Checks(unittest.TestCase):
    def test_pinned_environment_and_evaluator_sources(self):
        import importlib.metadata
        import nuscenes
        import sys
        self.assertEqual(sys.version_info[:3], (3, 10, 22))
        for name, version in {'nuscenes-devkit': '1.2.0', 'numpy': '1.26.4', 'scipy': '1.15.3',
                              'matplotlib': '3.8.4', 'scikit-learn': '1.7.2', 'shapely': '2.0.7'}.items():
            self.assertEqual(importlib.metadata.version(name), version)
        sdk = Path(nuscenes.__file__).parent
        for name, digest in {
            'eval/detection/evaluate.py': '1547188122d60ce0bc8600cb27861b10f624e3d38dbfb08e5569ffebb6cbba6e',
            'eval/detection/algo.py': '460e1691c1391e01fe7c6f4576add6054b29cc280687513a9f9737b7363f4f38',
            'eval/common/loaders.py': '9853cb47dd898c85e2032507e6bb419460f173e6b9a3e069675efe0839e72fe7',
            'utils/splits.py': 'eab6fa5e2536a2a85bd9451fb35771833e262b4b96319a6b26fee1dce8f4e2cd'}.items():
            self.assertEqual(e.file_sha(sdk / name), digest)

    def test_comparison_detects_errors_and_handles_nan(self):
        expected = {'score': [0.5, float('nan')], 'eval_time': 10}
        self.assertEqual(e.compare_metrics({'score': [0.5, float('nan')], 'eval_time': 99}, expected)[0], [])
        self.assertTrue(e.compare_metrics({'score': [0.6, float('nan')], 'eval_time': 99}, expected)[0])
        self.assertTrue(e.compare_metrics({'score': [0.5, 0], 'eval_time': 99}, expected)[0])
        self.assertTrue(e.compare_metrics({'score': [0.5], 'eval_time': 99}, expected)[0])

    def test_empty_frames_and_order_retained(self):
        from nuscenes.eval.common.data_classes import EvalBoxes
        boxes = EvalBoxes()
        for t in ('c', 'a', 'b'):
            boxes.add_boxes(t, [])
        subset = e.subset_boxes(boxes, {'c', 'b'})
        self.assertEqual(subset.sample_tokens, ['c', 'b'])
        self.assertEqual(boxes.sample_tokens, ['c', 'a', 'b'])
        with self.assertRaises(AssertionError): e.subset_boxes(boxes, {'missing'})

    def test_lean_metadata_full_replay_and_report(self):
        from nuscenes import NuScenes
        from nuscenes.eval.common.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
        import csv
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); source = d / 'source'; target = d / 'lean'
            expected = tables(source)
            import io, tarfile
            tgz = io.BytesIO()
            with tarfile.open(fileobj=tgz, mode='w:gz') as tar:
                for name in expected:
                    tar.add(source / name, arcname='./' + name)
            archive = d / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('metadata/v1.0-trainval_meta.tgz', tgz.getvalue())
            restored = d / 'restored'
            r.restore_metadata(restored, archive, expected)
            r.restore_metadata(restored, d / 'does_not_exist.zip', expected)
            for name, record in expected.items():
                self.assertEqual(e.file_sha(restored / name), record['sha256'])
            (restored / 'v1.0-trainval/category.json').write_text('corrupt')
            with self.assertRaises(AssertionError):
                r.restore_metadata(restored, archive, expected)
            derivation = r.derive_evaluation_metadata(source, target, {'a', 'b'}, expected)
            self.assertEqual(derivation['row_counts']['v1.0-trainval/sample_data.json']['retained_rows'], 2)
            self.assertEqual(r.file_sha(source / 'v1.0-trainval/sample_annotation.json'),
                             r.file_sha(target / 'v1.0-trainval/sample_annotation.json'))
            data = {t: [{'sample_token': t, 'translation': [10 + i, 0, 0], 'size': [2, 4, 2],
                    'rotation': [1, 0, 0, 0], 'velocity': [2, 0], 'detection_name': 'car',
                    'detection_score': 0.9, 'attribute_name': 'vehicle.parked'}] for i, t in enumerate('ab')}
            prediction = d / 'predictions.json'
            prediction.write_text(json.dumps({'results': data, 'meta': {'use_lidar': True}}))
            results = []
            for index, root in enumerate((source, target)):
                nusc = NuScenes(version='v1.0-trainval', dataroot=str(root), verbose=False)
                # Temporal annotation outside chosen sensor rows remains available.
                self.assertEqual(list(nusc.box_velocity('ab')), [2., 0., 0.])
                evaluator = NuScenesEval(nusc, config_factory('detection_cvpr_2019'), str(prediction),
                                        'val', str(d / f'init{index}'), verbose=False)
                summary, coverage = e.save_evaluation(evaluator, d / f'metrics{index}')
                self.assertAlmostEqual(summary['mean_ap'], 0.1, places=12)
                self.assertEqual(len(coverage['absent_gt_classes']), 9)
                results.append(summary)
            self.assertEqual(e.compare_metrics(*results)[0], [])
            # Exercise the final paired report with deliberately sparse GT coverage.
            work = d / 'report'; (work / 'preparation').mkdir(parents=True)
            rows = [{'scene_name': f'scene-{i}', 'group': g, 'log_token': 'same_log'}
                    for i, g in enumerate(e.GROUPS)]
            with (work / 'preparation/scene_inventory.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            for model in e.MODELS:
                (work / model).mkdir()
                (work / model / 'completion.json').write_text(json.dumps({'full_replay_pass': True}))
                for group in e.GROUPS:
                    group_dir = work / model / group; group_dir.mkdir()
                    (group_dir / 'metrics_summary.json').write_text(json.dumps(results[0]))
                    (group_dir / 'coverage.json').write_text(json.dumps(coverage))
            e.build_report(work)
            paired = json.loads((work / 'paired_summary.json').read_text())
            self.assertEqual(paired['common_GT_classes'], ['car'])
            self.assertTrue((work / 'weather_group_comparison.png').stat().st_size > 10000)
            # Keep a synthetic figure for layout review, with an explicit synthetic name.
            import shutil
            shutil.copyfile(work / 'weather_group_comparison.png', '/tmp/weather_synthetic_report.png')


if __name__ == '__main__':
    unittest.main()
