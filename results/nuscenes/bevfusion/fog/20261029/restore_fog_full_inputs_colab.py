"""Restore and content-verify the unchanged 600-frame fog inputs; no inference.

Each selected sensor payload must match its previously preserved SHA-256 from
an immutable Git-pinned 9,600-file restoration audit. No whole-archive rehash.
Reuses valid local files after interruptions; never changes the population.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback
import urllib.request
import zipfile

BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
PRIOR = Path('/content/fog_gpu_smoke_20261029/20261029T080852328937Z')
WORK = Path('/content/fog_full_inputs_20261029')
REF = '90b7dcc069970a8d60e2917b39d90e8114ecd533'
AUDIT_PATH = 'results/nuscenes/bevfusion/camera_availability/20261014/original_evidence/sensor_restoration_audit.json'
AUDIT_BLOB = '2b377de0008f08a0f7a2bb2fe20121ec2418ff50'
STRICT_ZIP_SHA = 'a3127d6776fd175d5466a3d7272234efcdf3cd3df0aa60cff12d5ec739b3ba6d'
PREFLIGHT_SCRIPT_SHA = 'c919c0f36579d6026315a302909593076b69a19e6b64274039a0228fb9cd5bf7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def verify_zip(path, expected, manifest_name):
    assert sha(path) == expected
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        files = json.loads(z.read(manifest_name))['files']
        assert set(z.namelist()) == set(files) | {manifest_name}
        for name, record in files.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        return json.loads(z.read('paired_smoke_audit.json'))


def restore_archive(archive, expected_archive_bytes, expected, root, indexer):
    """Selected-payload verification is anchored to external pinned hashes."""
    assert archive.stat().st_size == expected_archive_bytes, 'Archive size differs: ' + archive.name
    records = {}; restored = reused = 0
    with zipfile.ZipFile(archive) as z:
        index = indexer(z.namelist(), set(expected))
        assert set(index) == set(expected), 'Missing/ambiguous selected source members'
        for relative in sorted(index, key=lambda p: z.getinfo(index[p]).header_offset):
            record = expected[relative]; target = root / relative
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            if target.exists():
                assert target.is_file() and target.stat().st_size == record['bytes'] and sha(target) == record['sha256'], 'Existing local sensor hash mismatch: ' + relative
                reused += 1
            else:
                info = z.getinfo(index[relative]); assert info.file_size == record['bytes']
                data = z.read(info)  # ZIP CRC checked in addition to the external SHA-256.
                assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], 'Source payload hash mismatch: ' + relative
                if relative.endswith('.pcd.bin'): assert len(data) % 20 == 0
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.fog_restore_partial')
                # An interrupted partial is never accepted as a complete input.
                with partial.open('wb') as f: f.write(data)
                assert sha(partial) == record['sha256']; partial.replace(target)
                restored += 1
            records[relative] = {'bytes': record['bytes'], 'sha256': record['sha256'],
                                 'source_archive': archive.name, 'source_member': index[relative]}
    return records, restored, reused


def backup(out, dest, stamp, success):
    records = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out / 'manifest.json', {'files': records})
    archive = out.parent / ('fog_full_input_restoration_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*records, 'manifest.json']: z.write(out / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in records.items():
            data = z.read(name); assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    target = dest / archive.name
    with archive.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    assert sha(archive) == sha(target)
    receipt = {'status': 'FOG_FULL_INPUT_RESTORATION_AND_DRIVE_COPY_PASS' if success else 'FOG_FULL_INPUT_RESTORATION_FAILED_EVIDENCE_PRESERVED',
        'archive': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target),
        'scope': '9600-input restoration audit only; no new corruption or detector inference.'}
    write(dest / 'backup_receipt.json', receipt); print(json.dumps(receipt, indent=2)); print('UPLOAD THIS ZIP:', target)


def main():
    assert BASE.is_dir() and PRIOR.is_dir(), 'Keep the current Colab session and project Drive mounted.'
    started = time.monotonic(); stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = WORK / 'runs' / stamp; out.mkdir(parents=True, exist_ok=False)
    root = WORK / 'dataset'; root.mkdir(parents=True, exist_ok=True)
    dest = BASE / 'fog_20261029/full_inputs' / stamp; dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    error = None
    try:
        smoke_zip = BASE / 'fog_20261029/gpu_smoke_deterministic/20261029T085752261528Z/fog_gpu_smoke_deterministic_20261029T085752261528Z.zip'
        smoke = verify_zip(smoke_zip, STRICT_ZIP_SHA, 'manifest.json')
        assert smoke['status'] == 'PAIRED_STRICT_DETERMINISTIC_FOG_GPU_SMOKE_PASS'
        assert smoke['original_positional_baseline_repeat_pass_each_model'] and smoke['same_packed_points_per_condition']
        helper = PRIOR / 'evidence/code/preflight_fog_colab.py'; assert sha(helper) == PREFLIGHT_SCRIPT_SHA
        spec = importlib.util.spec_from_file_location('frozen_fog_preflight', helper)
        preflight = importlib.util.module_from_spec(spec); spec.loader.exec_module(preflight)
        population = PRIOR / 'pilot/frozen_population'
        source = {}
        for relative, blob in preflight.PROJECT_FILES.items():
            data = (population / Path(relative).name).read_bytes()
            assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob
            source[Path(relative).name] = data
        tokens, inventory, required = preflight.validate_population(source['sample_tokens.txt'], source['selected_input_manifest.json'])
        assert len(required) == 9600 and len(tokens) == 600
        for name, data in source.items(): (out / name).write_bytes(data)
        protocol = {'status': 'FULL_FOG_INPUT_RESTORATION_PROTOCOL_PRESERVED_BEFORE_RESTORATION',
            'sample_count': 600, 'scene_count': 150, 'required_unique_files': 9600,
            'sample_tokens_sha256': hashlib.sha256(source['sample_tokens.txt']).hexdigest(),
            'strict_paired_smoke_zip_sha256': STRICT_ZIP_SHA, 'reference_commit': REF,
            'preserved_sensor_audit_git_blob': AUDIT_BLOB, 'script_sha256': sha(__file__),
            'verification': 'Fresh selected sensor-byte SHA-256 checks against an immutable pinned prior audit, ZIP CRC on newly read members, and local write/reuse hashes. Archive size checked; no fresh whole multi-GB archive hash claim.',
            'restoration_policy': 'Only the unchanged 9600 required paths, from each recorded source archive. Existing local files are reused only after matching expected byte count and SHA-256. No population substitution.',
            'no_inference': True, 'no_corruption_preparation': True, 'dataset_root': str(root)}
        write(out / 'protocol.json', protocol); shutil.copyfile(out / 'protocol.json', dest / 'protocol.json')
        assert sha(out / 'protocol.json') == sha(dest / 'protocol.json')
        url = f'https://raw.githubusercontent.com/abjain182002/multimodal-perception/{REF}/{AUDIT_PATH}'
        data = urllib.request.urlopen(url, timeout=180).read()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == AUDIT_BLOB
        (out / 'preserved_sensor_restoration_audit.json').write_bytes(data)
        frozen = json.loads(data); expected = frozen['members']
        assert set(expected) == required and frozen['verified_sensor_files'] == 9600
        assert sum(r['bytes'] for r in expected.values()) == frozen['sensor_bytes'] == 4700293684
        groups = defaultdict(dict)
        for relative, record in expected.items(): groups[record['source_archive']][relative] = record
        archives = {r['archive']: r for r in frozen['archives']}
        assert set(groups) == set(archives)
        remaining = sum(r['bytes'] for name, r in expected.items() if not (root / name).is_file())
        free = shutil.disk_usage(root).free
        assert free >= remaining + 2 * 1024**3, f'Insufficient local disk: need {remaining} bytes plus 2 GiB audit/headroom; free={free}'
        records = {}; restored = reused = 0
        for name in sorted(groups):
            print('RESTORING/HASH-CHECKING SELECTED INPUTS:', name, len(groups[name]), flush=True)
            result, new, old = restore_archive(BASE / name, archives[name]['bytes'], groups[name], root, preflight.sensor_index)
            records.update(result); restored += new; reused += old
            write(out / ('archive_' + name + '.json'), {'archive': name, 'selected_files_verified': len(result),
                'restored': new, 'reused': old, 'whole_archive_freshly_rehashed': False,
                'historical_archive_sha256': archives[name]['sha256']})
            print('VERIFIED INPUTS:', len(records), '/', len(required), flush=True)
        assert set(records) == required and len(records) == 9600
        annotation = PRIOR / 'annotations_kaggle.pkl'
        assert sha(annotation) == '4f11db818b8d5504fc2bb432fcd51a28b09fec8d0b9c4063f2a017413fbfa99d'
        report = {'status': 'FOG_FULL_600_FRAME_SENSOR_CONTENT_RESTORATION_PASS',
            'samples': 600, 'scenes': 150, 'verified_sensor_files': len(records), 'sensor_bytes': sum(r['bytes'] for r in records.values()),
            'dataset_root': str(root), 'members': records, 'newly_restored': restored, 'reused': reused,
            'annotation_source': str(annotation), 'annotation_source_sha256': sha(annotation),
            'annotation_unpickled': False, 'local_disk_free_bytes_after_restoration': shutil.disk_usage(root).free,
            'elapsed_seconds': round(time.monotonic() - started, 2),
            'scope': 'Every selected sensor payload matches its prior preserved SHA-256; archive size/ZIP CRC checked, no fresh complete-archive hashes. No fog caches, new detector inference, AP/NDS or complete fog accuracy claim.'}
        write(out / 'restoration_audit.json', report)
        success = True
    except BaseException:
        error = traceback.format_exc(); write(out / 'failure.json', {'traceback': error}); success = False
    backup(out, dest, stamp, success)
    if error: raise RuntimeError('Input restoration stopped; evidence preserved. Valid restored files remain reusable.')


if __name__ == '__main__': main()
