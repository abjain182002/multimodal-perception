"""Create and verify a frozen four-frame KITTI validation inference package."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


FRAME_IDS = ('000831', '001662', '003324', '004155')
EXTENSIONS = {'calib': '.txt', 'image_2': '.png', 'label_2': '.txt', 'velodyne': '.bin'}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def entry(name, length):
    info = tarfile.TarInfo(name)
    info.size = length
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mode = 0o644
    return info


def verify(archive_path, manifest):
    expected = {'manifest.json': hashlib.sha256((json.dumps(manifest, indent=2) + '\n').encode()).hexdigest(),
                **{'training/' + part + '/' + fid + suffix: manifest['input_sha256'][fid][part]
                   for fid in FRAME_IDS for part, suffix in EXTENSIONS.items()},
                'splits/train.txt': manifest['train_split_sha256'],
                'splits/val.txt': manifest['validation_split_sha256']}
    seen = set()
    with tarfile.open(archive_path, 'r:gz') as bundle:
        for member in bundle:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f'Unexpected or duplicate archive member: {member.name}')
            digest = hashlib.sha256()
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f'Cannot read {member.name}')
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != expected[member.name]:
                raise ValueError(f'Archive member hash mismatch: {member.name}')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError(f'Missing archive members: {sorted(set(expected) - seen)}')
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, default=Path('results/clocs_validation_input_preflight.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-output', type=Path, default=Path('results/clocs_val_smoke_4_manifest.json'))
    args = parser.parse_args()
    audit = json.loads(args.preflight.read_text())
    train_ids = set(args.train.read_text().split())
    val_ids = set(args.val.read_text().split())
    if (audit['status'] != 'PASS' or set(FRAME_IDS) - val_ids or set(FRAME_IDS) & train_ids
            or sha256(args.train) != audit['train_split_sha256']
            or sha256(args.val) != audit['val_split_sha256']):
        raise ValueError('Validation IDs or frozen split differ from the preflight')
    paths, hashes = {}, {}
    for fid in FRAME_IDS:
        paths[fid] = {}
        hashes[fid] = {}
        for part, suffix in EXTENSIONS.items():
            path = args.dataset / part / (fid + suffix)
            hashes[fid][part] = sha256(path)
            paths[fid][part] = path
    manifest = dict(schema_version='clocs_validation_smoke_inputs_v1', split='validation',
                    interpretation='Four validation frames for inference plumbing only; not a performance estimate.',
                    frame_ids=list(FRAME_IDS), train_split_sha256=sha256(args.train),
                    validation_split_sha256=sha256(args.val), input_sha256=hashes,
                    fusion_checkpoint_sha256=audit['fusion_checkpoint_sha256'],
                    pointpillars_checkpoint_sha256=audit['baseline_checkpoint_sha256'])
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.manifest_output.exists():
        raise FileExistsError('Refusing to overwrite an existing package or manifest')
    with tempfile.NamedTemporaryFile(prefix='.clocs_val_', suffix='.tar.gz',
                                     dir=args.output.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with staging.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w') as bundle:
                    bundle.addfile(entry('manifest.json', len(manifest_bytes)), io.BytesIO(manifest_bytes))
                    for name, path in (('splits/train.txt', args.train), ('splits/val.txt', args.val)):
                        with path.open('rb') as source:
                            bundle.addfile(entry(name, path.stat().st_size), source)
                    for fid in FRAME_IDS:
                        for part, suffix in EXTENSIONS.items():
                            path = paths[fid][part]
                            with path.open('rb') as source:
                                bundle.addfile(entry(f'training/{part}/{fid}{suffix}', path.stat().st_size), source)
        count = verify(staging, manifest)
        os.replace(staging, args.output)
    finally:
        staging.unlink(missing_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_bytes(manifest_bytes)
    print(json.dumps(dict(status='PASS', frames=len(FRAME_IDS), archive_members_verified=count,
                          archive_sha256=sha256(args.output), archive_bytes=args.output.stat().st_size,
                          archive=str(args.output), manifest=str(args.manifest_output)), indent=2))


if __name__ == '__main__':
    main()
