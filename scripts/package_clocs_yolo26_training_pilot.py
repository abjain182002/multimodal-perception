"""Verify and package the 16 YOLO26 training examples for GPU transfer."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

import numpy as np

from fusion.clocs_dataset import load_training_example_npz
from scripts.export_clocs_dense_train_frame import file_sha256
from scripts.build_clocs_yolo_train_frame import CAMERA_WEIGHT_SHA256


def collect_files(data_dir, summary_path):
    summary = json.loads(summary_path.read_text())
    records = summary['frames']
    if (summary['schema_version'] != 'clocs_yolo26_train_16_training_example_audit_v1'
            or summary['status'] != 'PASS' or summary['completed_frames'] != 16
            or len(records) != 16 or len({r['frame_id'] for r in records}) != 16
            or summary['yolo26x_weights_sha256'] != CAMERA_WEIGHT_SHA256['yolo26x']):
        raise ValueError('Expected completed, hash-pinned 16-frame YOLO26 pilot')
    files = {'summary.json': summary_path}
    totals = {key: 0 for key in summary['totals']}
    for record in records:
        fid = record['frame_id']
        if len(fid) != 6 or not fid.isdigit():
            raise ValueError('Invalid frame identifier')
        target = data_dir / f'{fid}_real_targets.npz'
        rows_path = data_dir / f'{fid}_clocs_rows.npz'
        audit_path = summary_path.parent / f'clocs_yolo26_frame_{fid}_audit.json'
        audit = json.loads(audit_path.read_text())
        if (file_sha256(audit_path) != record['frame_audit_sha256']
                or file_sha256(rows_path) != record['rows_sha256']
                or audit['camera_model'] != 'yolo26x' or audit['split'] != 'train'
                or audit['frame_id'] != fid
                or audit['source_sha256']['yolo26x_weights'] != CAMERA_WEIGHT_SHA256['yolo26x']
                or audit['source_sha256']['dense_candidates'] != record['candidate_sha256']
                or audit['source_sha256']['feature_rows'] != record['rows_sha256']
                or audit['training_example_content_sha256'] != record['training_example_content_sha256']):
            raise ValueError(f'Artifact provenance mismatch: {fid}')
        example = load_training_example_npz(target, verify_integrity=True)
        if (example['frame_id'] != fid or example['split'] != 'train'
                or example['content_sha256'] != record['training_example_content_sha256']):
            raise ValueError(f'Target mismatch: {fid}')
        with np.load(rows_path, allow_pickle=False) as rows:
            if (str(rows['camera_model'].item()) != 'yolo26x'
                    or str(rows['camera_weights_sha256'].item()) != CAMERA_WEIGHT_SHA256['yolo26x']
                    or not np.array_equal(rows['features'], example['feature_rows'])
                    or not np.array_equal(rows['candidate_index'], example['candidate_indices'])):
                raise ValueError(f'Feature/target mismatch: {fid}')
        for key in ('num_candidates', 'num_positive', 'num_negative', 'num_ignored'):
            if example[key] != record[key]:
                raise ValueError(f'Count mismatch: {fid} {key}')
        if example['num_rows'] != record['num_feature_rows']:
            raise ValueError(f'Row count mismatch: {fid}')
        for key in totals:
            totals[key] += int(record[key])
        files[f'examples/{target.name}'] = target
        files[f'rows/{rows_path.name}'] = rows_path
        files[f'audits/{audit_path.name}'] = audit_path
    if totals != summary['totals']:
        raise ValueError('Summary totals differ from frame records')
    return files, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path,
                        default=Path('results/kitti/clocs/training_exports/clocs_yolo26_train_16_training_example_audit.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Archive already exists: {args.output}')
    files, totals = collect_files(args.data_dir, args.summary)
    manifest = {
        'schema_version': 'clocs_yolo26_training_transfer_v1',
        'camera_model': 'yolo26x', 'split': 'train', 'num_frames': 16,
        'interpretation': 'Training pilot only; not held-out evaluation.',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'totals': totals,
        'files': {name: {'sha256': file_sha256(path), 'bytes': path.stat().st_size}
                  for name, path in files.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        archive = Path(temporary) / 'pilot.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            raw = (json.dumps(manifest, indent=2) + '\n').encode()
            info = tarfile.TarInfo('manifest.json')
            info.size = len(raw)
            bundle.addfile(info, io.BytesIO(raw))
            for name, path in files.items():
                bundle.add(path, arcname=name, recursive=False)
        with tarfile.open(archive, 'r:gz') as bundle:
            if set(bundle.getnames()) != set(files) | {'manifest.json'}:
                raise ValueError('Unexpected archive members')
            for name, entry in manifest['files'].items():
                data = bundle.extractfile(name).read()
                if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError(f'Archive verification failed: {name}')
        # Exclusive creation prevents replacing an existing transfer archive.
        with args.output.open('xb') as destination, archive.open('rb') as source:
            import shutil
            shutil.copyfileobj(source, destination)
    print(json.dumps({'status': 'PACKAGE_PASS', 'num_frames': 16,
                      'verified_members': len(files), 'totals': totals,
                      'archive': str(args.output), 'archive_bytes': args.output.stat().st_size,
                      'archive_sha256': file_sha256(args.output)}, indent=2))


if __name__ == '__main__':
    main()

