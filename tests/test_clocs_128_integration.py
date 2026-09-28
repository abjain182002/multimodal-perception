"""Synthetic CPU integration test; does not establish real KITTI accuracy."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from fusion.clocs_dataset import build_training_example, save_training_example_npz, training_example_content_sha256
from fusion.clocs_sampling import PILOT_IDS
from scripts.train_clocs_pilot import sha256


class IntegrationTest(unittest.TestCase):
    def test_complete_runs_and_input_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('examples', 'rows', 'audits'):
                (root / folder).mkdir()
            ids = sorted(PILOT_IDS | {f'{i:06}' for i in range(8000, 8112)})
            train, val = root / 'train.txt', root / 'val.txt'
            train.write_text('\n'.join(ids) + '\n')
            val.write_text('009999\n')
            totals = dict(car_gt_after_range_filter=128, yolo26x_car_detections=128,
                          num_candidates=512, num_feature_rows=640, num_positive=128,
                          num_negative=256, num_ignored=128)
            for fid in ids:
                features = np.array([[.8,.9,.9,.1],[.1,.7,.8,.3],[-10,.2,-10,.5],[.5,.6,.7,.2],[.2,.7,.4,.3]], dtype=np.float32)
                indices = np.array([0,1,2,3,1], dtype=np.int64)
                example = build_training_example(fid, 'train', features, indices,
                                                np.array([.9,.7,.2,.6], dtype=np.float32),
                                                np.array([.8,.1,0,.6], dtype=np.float32))
                save_training_example_npz(example, root / 'examples' / f'{fid}_real_targets.npz')
                rows = root / 'rows' / f'{fid}_clocs_rows.npz'
                np.savez_compressed(rows, features=features, candidate_index=indices, camera_model='yolo26x')
                audit = {key: value // 128 for key, value in totals.items()}
                audit.update(frame_id=fid, split='train', training_example_content_sha256=training_example_content_sha256(example),
                    source_sha256=dict(feature_rows=sha256(rows), split_manifest='fixture',
                    yolo26x_weights='9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'))
                (root / 'audits' / f'clocs_yolo26_frame_{fid}_audit.json').write_text(json.dumps(audit))
            (root / 'clocs_yolo26_train_128_summary.json').write_text(json.dumps(dict(status='PASS', completed_frames=128, frame_ids=ids, totals=totals)))
            manifest = root / 'batch.json'
            manifest.write_text(json.dumps(dict(split='train', frames=[dict(frame_id=f) for f in ids],
                train_split_sha256=sha256(train), validation_split_sha256=sha256(val), split_manifest_sha256='fixture')))
            common = [sys.executable, '-m', 'scripts.train_clocs_128', '--data-dir', str(root),
                      '--batch-manifest', str(manifest), '--train-ids', str(train), '--val-ids', str(val),
                      '--epochs', '1', '--device', 'cpu']
            for ratio in (0, 8):
                out = root / f'run{ratio}'
                command = common + ['--negative-ratio', str(ratio), '--output-dir', str(out)]
                import os
                run = subprocess.run(command, text=True, capture_output=True,
                                     env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / 'report.json').read_text())
                self.assertEqual(report['status'], 'PASS')
                self.assertEqual(report['reload_ap_absolute_error'], 0)
                self.assertEqual(report['best_dev']['evaluated_candidates'], 32 * 3)
                retry = subprocess.run(command, text=True, capture_output=True)
                self.assertNotEqual(retry.returncode, 0)
            np.savez(root / 'rows' / f'{ids[0]}_clocs_rows.npz', features=np.zeros((1,4)))
            bad = subprocess.run(common + ['--output-dir', str(root / 'bad')], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
