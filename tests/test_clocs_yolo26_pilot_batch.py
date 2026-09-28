import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion.clocs_dataset import build_training_example, save_training_example_npz
from scripts.build_clocs_yolo26_pilot_batch import inspect_example
from scripts.export_clocs_dense_train_frame import file_sha256


class TestYOLO26PilotResume(unittest.TestCase):
    def test_resume_checks_camera_and_exact_feature_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n = 107136
            features = np.tile(np.array([-10, .5, -10, .1], np.float32), (n, 1))
            indices = np.arange(n)
            example = build_training_example('000000', 'train', features, indices,
                                             np.full(n, .5), np.zeros(n))
            target = root / 'targets.npz'
            saved = save_training_example_npz(example, target)
            rows = root / 'rows.npz'
            def write_rows(values):
                np.savez_compressed(rows, frame_id='000000', camera_model='yolo26x',
                                    camera_weights_sha256='yolo', features=values,
                                    candidate_index=indices)
            write_rows(features)
            frame = {'frame_id': '000000', 'stratum': 'zero',
                     'car_gt_after_range_filter': 0,
                     'input_sha256': {'image_2': 'image', 'calib': 'calib', 'label_2': 'label'}}
            audit = {
                'schema_version': 'clocs_real_train_frame_audit_v2',
                'frame_id': '000000', 'split': 'train', 'camera_model': 'yolo26x',
                'camera_confidence_threshold': .25, 'ultralytics_version': '8.4.132',
                'training_example_content_sha256': saved['content_sha256'],
                'num_feature_rows': n, 'num_positive': 0, 'num_negative': n,
                'num_ignored': 0, 'best_iou_nonzero': 0, 'best_iou_max': 0,
                'car_gt_after_range_filter': 0, 'yolo26x_car_detections': 0,
                'source_sha256': {
                    'dense_candidates': 'dense', 'image': 'image', 'calibration': 'calib',
                    'label': 'label', 'yolo26x_weights': 'yolo', 'split_manifest': 'split',
                    'pointpillars_checkpoint': 'd42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e',
                    'pointpillars_config': '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc',
                    'feature_rows': file_sha256(rows),
                },
            }
            audit_path = root / 'audit.json'
            def inspect():
                audit_path.write_text(json.dumps(audit))
                return inspect_example(frame, audit_path, rows, target, 'dense', {}, 'yolo', 'split')
            self.assertEqual(inspect()['num_negative'], n)
            audit['camera_model'] = 'yolov8x'
            with self.assertRaisesRegex(ValueError, 'Camera protocol'):
                inspect()
            audit['camera_model'] = 'yolo26x'
            changed = features.copy()
            changed[0, 1] = .7
            write_rows(changed)
            audit['source_sha256']['feature_rows'] = file_sha256(rows)
            with self.assertRaisesRegex(ValueError, 'feature rows differ'):
                inspect()


if __name__ == '__main__':
    unittest.main()
