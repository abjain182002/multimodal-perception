#!/usr/bin/env python3
"""Build and audit 16 frozen train-only YOLO26x+CLOCs examples on Mac."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fusion.clocs_dataset import load_training_example_npz
from scripts.export_clocs_dense_train_frame import file_sha256
from scripts.validate_kitti_split import read_frame_ids, validate_split


def inspect_example(frame, audit_file, rows_file, example_file, candidate_sha,
                    source_paths, yolo_sha, split_sha):
    audit = json.loads(audit_file.read_text(encoding='utf-8'))
    fid = frame['frame_id']
    if audit['schema_version'] != 'clocs_real_train_frame_audit_v2' or audit['frame_id'] != fid or audit['split'] != 'train':
        raise ValueError(f'Unexpected audit for {fid}')
    if (audit.get('camera_model') != 'yolo26x'
            or audit.get('camera_confidence_threshold') != 0.25
            or audit.get('ultralytics_version') != '8.4.132'):
        raise ValueError(f'Camera protocol changed for {fid}')
    expected_hashes = {
        'dense_candidates': candidate_sha,
        'image': frame['input_sha256']['image_2'],
        'calibration': frame['input_sha256']['calib'],
        'label': frame['input_sha256']['label_2'],
        'yolo26x_weights': yolo_sha,
        'split_manifest': split_sha,
        'pointpillars_checkpoint': 'd42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e',
        'pointpillars_config': '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc',
        'feature_rows': file_sha256(rows_file),
    }
    for key, expected in expected_hashes.items():
        if audit['source_sha256'][key] != expected:
            raise ValueError(f'Unexpected {key} hash for {fid}')
    example = load_training_example_npz(example_file, verify_integrity=True)
    if (example['frame_id'] != fid or example['split'] != 'train'
            or example['content_sha256'] != audit['training_example_content_sha256']):
        raise ValueError(f'Training example does not match audit for {fid}')
    with np.load(rows_file, allow_pickle=False) as rows:
        if (str(rows['frame_id'].item()) != fid
                or str(rows['camera_model'].item()) != 'yolo26x'
                or str(rows['camera_weights_sha256'].item()) != yolo_sha
                or not np.array_equal(rows['features'], example['feature_rows'])
                or not np.array_equal(rows['candidate_index'], example['candidate_indices'])):
            raise ValueError(f'YOLO feature rows differ from training example for {fid}')
    if (example['num_candidates'] != 107136
            or sum(example[key] for key in ('num_positive', 'num_negative', 'num_ignored')) != 107136
            or example['num_rows'] != audit['num_feature_rows']
            or example['num_positive'] != audit['num_positive']
            or example['num_negative'] != audit['num_negative']
            or example['num_ignored'] != audit['num_ignored']
            or int(np.count_nonzero(example['best_iou'])) != audit['best_iou_nonzero']
            or int(audit['car_gt_after_range_filter']) != frame['car_gt_after_range_filter']):
        raise ValueError(f'Counts or GT alignment changed for {fid}')
    if frame['car_gt_after_range_filter'] == 0 and (
            example['num_positive'] != 0 or audit['best_iou_nonzero'] != 0):
        raise ValueError(f'Empty-GT control has positive IoU: {fid}')
    for field, path in source_paths.items():
        if file_sha256(path) != frame['input_sha256'][field]:
            raise ValueError(f'KITTI input changed: {path}')
    return {
        'frame_id': fid, 'stratum': frame['stratum'],
        'car_gt_after_range_filter': int(audit['car_gt_after_range_filter']),
        'yolo26x_car_detections': int(audit['yolo26x_car_detections']),
        'num_candidates': example['num_candidates'], 'num_feature_rows': example['num_rows'],
        'num_positive': example['num_positive'], 'num_negative': example['num_negative'],
        'num_ignored': example['num_ignored'], 'best_iou_nonzero': audit['best_iou_nonzero'],
        'best_iou_max': float(audit['best_iou_max']),
        'candidate_sha256': candidate_sha,
        'rows_sha256': file_sha256(rows_file),
        'training_example_content_sha256': example['content_sha256'],
        'frame_audit_sha256': file_sha256(audit_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--candidates-dir', required=True)
    parser.add_argument('--yolo-weights', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--audit-dir', default='results')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--manifest', default='configs/splits/kitti_second_split.json')
    parser.add_argument('--batch', default='results/clocs_train_export_batch_manifest.json')
    parser.add_argument('--receipt', default='results/clocs_train_16_dense_export_stdout_receipt.json')
    parser.add_argument('--summary', default='results/clocs_yolo26_train_16_training_example_audit.json')
    args = parser.parse_args()

    from importlib.metadata import version
    if version('ultralytics') != '8.4.132':
        raise ValueError('Use frozen Ultralytics 8.4.132 for this pilot')

    dataset = Path(args.dataset)
    outputs = Path(args.output_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    batch = json.loads(Path(args.batch).read_text(encoding='utf-8'))
    receipt = json.loads(Path(args.receipt).read_text(encoding='utf-8'))
    if (batch['schema_version'] != 'clocs_train_export_batch_v1'
            or batch['split'] != 'train' or len(batch['frames']) != 16
            or receipt['schema_version'] != 'clocs_train_16_dense_export_stdout_receipt_v1'
            or receipt['num_frames'] != 16
            or receipt['status'] != 'REPORTED_PASS'
            or file_sha256(args.batch) != receipt['batch_manifest_sha256']
            or [f['frame_id'] for f in batch['frames']] != [f['frame_id'] for f in receipt['frames']]):
        raise ValueError('Batch manifest differs from verified Colab export receipt')
    split = validate_split(args.manifest, args.train, args.val)
    if not split['ok'] or file_sha256(args.manifest) != batch['split_manifest_sha256']:
        raise ValueError('Frozen split changed')
    train_ids = set(read_frame_ids(args.train))
    if any(f['frame_id'] not in train_ids for f in batch['frames']):
        raise ValueError('A selected frame is outside the frozen training split')
    yolo_sha = file_sha256(args.yolo_weights)
    if yolo_sha != '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92':
        raise ValueError('YOLO26x weights changed')

    records = []
    for frame, reference in zip(batch['frames'], receipt['frames']):
        fid = frame['frame_id']
        candidate_file = Path(args.candidates_dir) / f'{fid}_dense_candidates.npz'
        if file_sha256(candidate_file) != reference['candidate_sha256']:
            raise ValueError(f'Dense candidate changed: {fid}')
        paths = {field: dataset / relative for field, relative in frame['input_files'].items()}
        for field, path in paths.items():
            if file_sha256(path) != frame['input_sha256'][field]:
                raise ValueError(f'KITTI input changed: {path}')
        example_file = outputs / f'{fid}_real_targets.npz'
        rows_file = outputs / f'{fid}_clocs_rows.npz'
        audit_file = audit_dir / f'clocs_yolo26_frame_{fid}_audit.json'
        if all(p.is_file() for p in (example_file, rows_file, audit_file)):
            action = 'REUSED'
        else:
            if any(p.exists() for p in (example_file, rows_file, audit_file)):
                raise ValueError(f'Partial artifacts require inspection before retry: {fid}')
            action = 'BUILT'
            command = [
                sys.executable, '-m', 'scripts.build_clocs_yolo_train_frame',
                '--camera-model', 'yolo26x',
                '--dataset', str(dataset), '--frame-id', fid,
                '--candidates', str(candidate_file),
                '--yolo-weights', str(args.yolo_weights), '--device', args.device,
                '--train', str(args.train), '--val', str(args.val),
                '--manifest', str(args.manifest),
                '--output', str(example_file), '--rows', str(rows_file),
                '--audit', str(audit_file),
            ]
            print(f'START {fid}: YOLO26x + candidate targets', flush=True)
            run = subprocess.run(command, capture_output=True, text=True,
                                 env={**os.environ, 'MPLBACKEND': 'Agg'})
            if run.returncode:
                print(run.stdout, flush=True)
                print(run.stderr, file=sys.stderr, flush=True)
                raise RuntimeError(f'Frame {fid} target preparation failed ({run.returncode})')
        record = inspect_example(frame, audit_file, rows_file, example_file,
                                 reference['candidate_sha256'], paths, yolo_sha,
                                 batch['split_manifest_sha256'])
        record['action'] = action
        records.append(record)
        summary = {
            'schema_version': 'clocs_yolo26_train_16_training_example_audit_v1',
            'status': ('PASS' if len(records) == 16 and any(f['num_positive'] for f in records)
                       else 'BLOCKED_NO_POSITIVE_SIGNAL' if len(records) == 16 else 'IN_PROGRESS'),
            'interpretation': 'Train-only engineering pilot, not a trained fusion model or validation AP.',
            'batch_manifest_sha256': file_sha256(args.batch),
            'export_receipt_sha256': file_sha256(args.receipt),
            'yolo26x_weights_sha256': yolo_sha,
            'completed_frames': len(records),
            'totals': {key: sum(int(f[key]) for f in records) for key in (
                'car_gt_after_range_filter', 'yolo26x_car_detections', 'num_candidates',
                'num_feature_rows', 'num_positive', 'num_negative', 'num_ignored')},
            'frames_with_gt_but_no_positive_candidates': [
                f['frame_id'] for f in records if f['car_gt_after_range_filter'] > 0 and f['num_positive'] == 0
            ],
            'frames': records,
        }
        destination = Path(args.summary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.partial')
        temporary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        temporary.replace(destination)
        print(f"{len(records)}/16 {fid} {action} GT={record['car_gt_after_range_filter']} "
              f"YOLO={record['yolo26x_car_detections']} +{record['num_positive']} "
              f"-{record['num_negative']} ignored={record['num_ignored']}", flush=True)

    if not summary['totals']['num_positive']:
        raise ValueError('No positive training signal; investigate before training')
    print('TARGETS PASS: 16 real training examples with positive signal')
    print('Summary:', args.summary)


if __name__ == '__main__':
    main()

