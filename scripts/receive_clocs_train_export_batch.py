#!/usr/bin/env python3
"""Verify and install the 14 new train-only PointPillars candidates on Mac."""

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--receipt', default='results/kitti/clocs/training_exports/clocs_train_16_dense_export_stdout_receipt.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    receipt = json.loads(Path(args.receipt).read_text(encoding='utf-8'))
    if (receipt['schema_version'] != 'clocs_train_16_dense_export_stdout_receipt_v1'
            or receipt['status'] != 'REPORTED_PASS' or receipt['num_frames'] != 16
            or receipt['new_exports'] != 14 or receipt['known_controls_reused'] != 2):
        raise ValueError('Unexpected GitHub export receipt')
    all_frames = receipt['frames']
    if len(all_frames) != 16 or len({f['frame_id'] for f in all_frames}) != 16:
        raise ValueError('Duplicate or missing frame in receipt')
    new_frames = [f for f in all_frames if f['action'] == 'EXPORTED']
    controls = [f for f in all_frames if f['action'] == 'REUSED']
    if len(new_frames) != 14 or len(controls) != 2:
        raise ValueError('Unexpected exported/control frame counts')

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for control in controls:
        file = output / f"{control['frame_id']}_dense_candidates.npz"
        if digest_file(file) != control['candidate_sha256']:
            raise ValueError(f'Existing control candidate changed: {file}')

    expected_names = {'manifest.json', 'audit.json'} | {
        f"candidates/{frame['frame_id']}_dense_candidates.npz"
        for frame in new_frames
    }
    with tarfile.open(args.archive, 'r:') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (len(members) != len(expected_names)
                or set(names) != expected_names or not all(m.isfile() for m in members)):
            raise ValueError('Archive has missing, duplicate, or unexpected members')
        by_name = {member.name: member for member in members}
        if by_name['manifest.json'].size > 1024 * 1024 or by_name['audit.json'].size > 2 * 1024 * 1024:
            raise ValueError('Oversized metadata member')
        manifest_bytes = archive.extractfile(by_name['manifest.json']).read()
        audit_bytes = archive.extractfile(by_name['audit.json']).read()
        manifest = json.loads(manifest_bytes)
        audit = json.loads(audit_bytes)
        if (manifest['schema_version'] != 'clocs_train_14_candidate_transfer_v1'
                or manifest['audit_sha256'] != receipt['audit_sha256']
                or hashlib.sha256(audit_bytes).hexdigest() != receipt['audit_sha256']
                or audit['status'] != 'PASS' or audit['completed_frames'] != 16):
            raise ValueError('Colab audit does not match GitHub receipt')
        if (manifest['batch_manifest_sha256'] != receipt['batch_manifest_sha256']
                or [(f['frame_id'], f['action'], f['candidate_sha256'])
                    for f in audit['frames']]
                != [(f['frame_id'], f['action'], f['candidate_sha256'])
                    for f in all_frames]):
            raise ValueError('Colab audit frames differ from GitHub receipt')
        transfer = manifest['frames']
        if (len(transfer) != 14
                or [(f['frame_id'], f['candidate_sha256']) for f in transfer]
                != [(f['frame_id'], f['candidate_sha256']) for f in new_frames]):
            raise ValueError('Transfer manifest differs from GitHub receipt')

        installed = 0
        already_present = 0
        for frame in transfer:
            fid = frame['frame_id']
            name = f'{fid}_dense_candidates.npz'
            member = by_name[f'candidates/{name}']
            if member.size != frame['bytes'] or not (0 < member.size < 256 * 1024 * 1024):
                raise ValueError(f'Candidate size invalid: {fid}')
            target = output / name
            if target.exists():
                if digest_file(target) != frame['candidate_sha256']:
                    raise ValueError(f'Existing candidate differs: {target}')
                already_present += 1
                continue
            tmp_path = None
            try:
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=output, prefix=f'.{fid}_', suffix='.partial', delete=False) as temporary:
                    tmp_path = Path(temporary.name)
                    with archive.extractfile(member) as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                            temporary.write(chunk)
                if digest.hexdigest() != frame['candidate_sha256'] or tmp_path.stat().st_size != member.size:
                    raise ValueError(f'Transferred candidate checksum failed: {fid}')
                os.replace(tmp_path, target)
                installed += 1
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
        print(f'RECEIVE PASS: {installed} installed, {already_present} already present; 2 controls verified')
        print('Colab audit SHA-256:', receipt['audit_sha256'])
        print('Archive SHA-256:', digest_file(args.archive))
        print('Candidate directory:', output)


if __name__ == '__main__':
    main()

