#!/usr/bin/env python3
"""Export the verified train-only pilot with resumable Drive checkpoints."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.export_clocs_dense_train_frame import file_sha256
from scripts.validate_kitti_split import validate_split


def main():
    code = Path('/content/clocs_export_code')
    source = Path('/content/datasets/KITTI/training')
    split_root = Path('/content/datasets/KITTI/splits')
    drive = Path('/content/drive/MyDrive/Major Project/cloud transfer/clocs_stage5_checkpoints')
    batch_file = code / 'results/clocs_train_export_batch_manifest.json'
    reduced_file = drive / 'clocs_train_16_reduced_cloud_audit.json'
    split_file = code / 'configs/splits/kitti_second_split.json'
    train, val = split_root / 'train.txt', split_root / 'val.txt'
    model_root = Path('/content/mmdetection3d')
    config = model_root / 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    checkpoint = Path('/content/checkpoints/pointpillars_kitti_car.pth')
    report_file = drive / 'clocs_train_16_dense_export_audit.json'
    expected_controls = {
        '000000': '99076c37ebe03ed812ed422c5b57119ad5ba790ef9d537706084ef100de83d8a',
        '000149': '3943f256e48e1d9b072cf01f8a9165c3cff9f612b94125bd1986cf21668b1b99',
    }
    required_shapes = {
        'boxes_lidar': (107136, 7), 'boxes_camera': (107136, 7),
        'scores_3d': (107136,), 'direction_labels': (107136,),
        'projected_boxes': (107136, 4), 'distance_82': (107136,),
        'candidate_index': (107136,),
    }

    batch = json.loads(batch_file.read_text())
    reduced = json.loads(reduced_file.read_text())
    assert batch['schema_version'] == 'clocs_train_export_batch_v1'
    assert batch['split'] == 'train' and len(batch['frames']) == 16
    assert reduced['status'] == 'PASS' and reduced['num_frames'] == 16
    assert reduced['known_controls_passed'] == 2
    assert reduced['batch_manifest_sha256'] == file_sha256(batch_file)
    assert reduced['mmdetection3d_commit'] == batch['mmdetection3d_commit']
    assert file_sha256(split_file) == batch['split_manifest_sha256']
    assert file_sha256(train) == batch['train_split_sha256']
    assert file_sha256(val) == batch['validation_split_sha256']
    assert validate_split(split_file, train, val)['ok']
    assert subprocess.check_output(['git', '-C', str(model_root), 'rev-parse', 'HEAD'], text=True).strip() == batch['mmdetection3d_commit']
    assert file_sha256(config) == '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc'
    assert file_sha256(checkpoint) == batch['pointpillars_checkpoint_sha256']
    assert torch.cuda.is_available(), 'A Colab CUDA GPU is required'
    assert {f['frame_id'] for f in reduced['frames']} == {f['frame_id'] for f in batch['frames']}

    stage = Path('/content/datasets/KITTI/training_reduced')
    stage.mkdir(parents=True, exist_ok=True)
    for name, target in {'calib': source / 'calib', 'image_2': source / 'image_2',
                         'velodyne': source / 'velodyne_reduced'}.items():
        link = stage / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target, target_is_directory=True)
        assert link.is_symlink() and link.resolve() == target.resolve(), link

    def verify_npz(path, frame_id):
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved['frame_id'].item()) == frame_id
            assert str(saved['checkpoint_sha256'].item()) == batch['pointpillars_checkpoint_sha256']
            assert str(saved['config_sha256'].item()) == file_sha256(config)
            assert str(saved['split_manifest_sha256'].item()) == batch['split_manifest_sha256']
            for field, shape in required_shapes.items():
                assert saved[field].shape == shape, (frame_id, field, saved[field].shape)
            assert np.array_equal(saved['candidate_index'], np.arange(107136))
            scores = saved['scores_3d']
            assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
            return {'num_candidates': len(scores), 'score_max': float(scores.max())}

    records = []
    for frame in batch['frames']:
        fid = frame['frame_id']
        cloud = source / 'velodyne_reduced' / f'{fid}.bin'
        expected = next(f['reduced_sha256'] for f in reduced['frames'] if f['frame_id'] == fid)
        assert file_sha256(cloud) == expected, f'Cloud changed: {fid}'
        output = drive / f'{fid}_dense_candidates.npz'
        if output.is_file():
            details = verify_npz(output, fid)
            digest = file_sha256(output)
            if fid in expected_controls:
                assert digest == expected_controls[fid], f'Known candidate hash changed: {fid}'
            action = 'REUSED'
        else:
            assert fid not in expected_controls, f'Missing known control candidate: {fid}'
            temporary = drive / f'.{fid}_dense_candidates.partial.npz'
            temporary.unlink(missing_ok=True)
            command = [
                '/content/pointpillars310/bin/python', '-m', 'scripts.export_clocs_dense_train_frame',
                '--dataset', str(stage), '--frame-id', fid, '--config', str(config),
                '--checkpoint', str(checkpoint), '--train', str(train), '--val', str(val),
                '--manifest', str(split_file), '--evidence',
                str(code / 'results/pointpillars_evidence_manifest.json'), '--output', str(temporary),
            ]
            subprocess.run(command, check=True, cwd=code, env={**os.environ, 'MPLBACKEND': 'Agg'})
            details = verify_npz(temporary, fid)
            digest = file_sha256(temporary)
            temporary.replace(output)
            assert file_sha256(output) == digest
            action = 'EXPORTED'
        records.append({'frame_id': fid, 'action': action, 'candidate_sha256': digest,
                        'reduced_cloud_sha256': expected, **details})
        report = {
            'schema_version': 'clocs_train_16_dense_export_audit_v1',
            'status': 'PASS' if len(records) == 16 else 'IN_PROGRESS',
            'interpretation': 'Train-only engineering pilot; not validation AP or trained fusion.',
            'batch_manifest_sha256': file_sha256(batch_file),
            'reduced_cloud_audit_sha256': file_sha256(reduced_file),
            'checkpoint_sha256': batch['pointpillars_checkpoint_sha256'],
            'config_sha256': file_sha256(config),
            'completed_frames': len(records),
            'frames': records,
        }
        report_file.parent.mkdir(parents=True, exist_ok=True)
        partial_report = report_file.with_suffix('.json.partial')
        partial_report.write_text(json.dumps(report, indent=2) + '\n')
        partial_report.replace(report_file)
        print(f"{len(records)}/16 {fid} {action} {digest}", flush=True)

    print('DENSE EXPORT PASS: 16 verified training frames; 14 new, 2 controls reused')
    print('Audit saved:', report_file)


if __name__ == '__main__':
    main()
