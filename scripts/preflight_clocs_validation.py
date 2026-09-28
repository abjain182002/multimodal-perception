"""Check frozen KITTI validation inputs for a CLOCs detection evaluation."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_FUSION_SHA = 'a2203b9adee46b782e896b7f3b9344b5916433b5404d4bd871aa7e05d6b0c473'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='KITTI training/ folder containing image_2, velodyne, calib, label_2')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--fusion-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-evidence', type=Path, default=Path('results/pointpillars_evidence_manifest.json'))
    parser.add_argument('--split-manifest', type=Path, default=Path('configs/splits/kitti_second_split.json'))
    parser.add_argument('--train-batch', type=Path, default=Path('results/clocs_train_export_128_manifest.json'))
    parser.add_argument('--fusion-report', type=Path, default=Path('results/clocs_128_comparison/ratio0/report.json'))
    parser.add_argument('--output', type=Path, default=Path('results/clocs_validation_input_preflight.json'))
    args = parser.parse_args()
    evidence = json.loads(args.baseline_evidence.read_text())
    split_manifest = json.loads(args.split_manifest.read_text())
    batch = json.loads(args.train_batch.read_text())
    report = json.loads(args.fusion_report.read_text())
    train_ids = args.train.read_text().split()
    val_ids = args.val.read_text().split()
    if (len(train_ids) != 3712 or len(val_ids) != 3769
            or len(set(train_ids)) != len(train_ids)
            or len(set(val_ids)) != len(val_ids)
            or set(train_ids) & set(val_ids)):
        raise ValueError('Frozen KITTI train/validation split counts or disjointness failed')
    train_sha, val_sha = digest(args.train), digest(args.val)
    if (train_sha != split_manifest['train']['sha256']
            or val_sha != split_manifest['validation']['sha256']
            or train_sha != batch['train_split_sha256']
            or val_sha != batch['validation_split_sha256']):
        raise ValueError('KITTI split hashes differ from canonical split or 128-frame train manifest')
    if (not set(report['split']['train_frame_ids']) <= set(train_ids)
            or not set(report['split']['dev_frame_ids']) <= set(train_ids)):
        raise ValueError('Fusion optimization/development includes a nontraining frame')
    checkpoint_sha = digest(args.fusion_checkpoint)
    if (checkpoint_sha != EXPECTED_FUSION_SHA
            or checkpoint_sha != report['checkpoint_sha256']
            or report['best_epoch'] != 12):
        raise ValueError('Fusion checkpoint differs from selected 128-frame run')

    missing = []
    suffixes = {'image_2': '.png', 'velodyne': '.bin', 'calib': '.txt', 'label_2': '.txt'}
    for fid in val_ids:
        for folder, suffix in suffixes.items():
            if not (args.dataset / folder / (fid + suffix)).is_file():
                missing.append(f'{folder}/{fid}{suffix}')
    output = dict(schema_version='clocs_validation_input_preflight_v1',
                  status='PASS' if not missing else 'MISSING_KITTI_FILES',
                  interpretation='Validation input availability and frozen provenance; no fused detection AP measured.',
                  train_frames=len(train_ids), validation_frames=len(val_ids),
                  missing_validation_files=len(missing), missing_examples=missing[:20],
                  dataset_root=str(args.dataset.resolve()),
                  train_split_sha256=train_sha, val_split_sha256=val_sha,
                  baseline_evidence_train_sha256=evidence['dataset']['train_txt_sha256'],
                  baseline_evidence_val_sha256=evidence['dataset']['val_txt_sha256'],
                  baseline_evidence_val_hash_matches=(val_sha == evidence['dataset']['val_txt_sha256']),
                  fusion_checkpoint_sha256=checkpoint_sha,
                  fusion_checkpoint_path=str(args.fusion_checkpoint.resolve()),
                  baseline_checkpoint_sha256=evidence['model']['checkpoint_sha256'],
                  baseline_predictions_reported=evidence['predictions']['frames'],
                  baseline_prediction_file=evidence.get('prediction_file'),
                  pending=['Export validation candidate features using the frozen PointPillars and YOLO26x.',
                           'Apply this checkpoint to all candidates and match baseline postprocessing.',
                           'Compare 3D/BEV strict AP40 under the same KITTI evaluator.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    if missing:
        raise SystemExit(f'{len(missing)} required KITTI validation files are missing')


if __name__ == '__main__':
    main()
