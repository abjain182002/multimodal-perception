"""Inventory the frozen 600-frame inputs and recover one pilot on CPU.

Standard library only. No inference, corruption, package installation, model
loading or full multi-GB archive rehash. Never unpickle the supplied assets.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

PROJECT_REF = 'e190e4c02fd953c30c153ade5660ca4a7da95c13'
UPSTREAM_REF = '39652e35d10dd7c2aed0b71df8c8c7effe9e6349'
RAW = 'https://raw.githubusercontent.com/'
STUDY = 'results/nuscenes/bevfusion/camera_challenges/20261016/original_evidence/study/'
PROJECT_FILES = {
    STUDY + 'sample_tokens.txt': 'a22e6aabba7fe014802542b5177775e2586fd1cd',
    STUDY + 'selected_input_manifest.json': 'f6879add2ba9fe2b4f85b103902d10dacee50d0a',
}
UPSTREAM_FILES = {
    'LICENSE': 'ecb4c6d483c77f84c6507288d85433c85afe09a1',
    'converter/img.py': 'b1b96971c8d24a700da4d19cc31692f78183036f',
    'converter/img_converter.py': '8435ecc8b1b02b4ccd0dcddb9ab82fa8e5fa12d1',
    'converter/lidar.py': 'fb6baacc7600068d487f3c3d540ea3155a9c8ec4',
    'converter/lidar_converter.py': 'd8d6f7e98436bf392886d9dcf26985224e611ce3',
    'converter/fog_lookup_tables/integral_0m_to_200m_stepsize_0.1m_alpha_0.02_beta_0.008.pickle': 'b64fbda6e4bca62ed6b8252164baf8eb1a3cb5c6',
    'converter/fog_lookup_tables/integral_0m_to_200m_stepsize_0.1m_alpha_0.03_beta_0.008.pickle': 'e835509154f0a172f4a383db6da8dbf7137d83e1',
    'converter/fog_lookup_tables/integral_0m_to_200m_stepsize_0.1m_alpha_0.06_beta_0.05.pickle': '3ab832b81fb3bf8d9f028aefcb13bad19e9a1b0f',
}
PLAN = {
    'status': 'FOG_DESIGN_AND_PILOT_RULE_FIXED_IMPLEMENTATION_VALIDATION_PENDING',
    'question': 'How do the two existing pretrained pipelines degrade under a published synthetic camera/LiDAR fog challenge on the same frozen frames?',
    'population': 'Existing 600 frames / 150 scenes, same order as prior challenge studies. Do not select easier frames or change population based on outcomes.',
    'tokens_sha256': 'ffda09ad54f1381f8bd56d0d204ef72ce66d58a08563dcb4bbe8ca589c89164f',
    'models': ['lidar', 'lidar_camera'],
    'checkpoint_sha256': {'lidar': '2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e',
                          'lidar_camera': 'e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab'},
    'detector_source_commit': 'fe25f7a51d36e3702f961e198894580d83c4387b',
    'method': 'MultiCorrupt source implementation and lookup tables, pinned to ' + UPSTREAM_REF,
    'planned_conditions': ['unperturbed', 'fog_severity_1', 'fog_severity_2', 'fog_severity_3'],
    'planned_runs': '2 fresh unperturbed baselines + 6 fog runs. Original scenes may already contain natural rain/night; unperturbed does not mean clear weather.',
    'lidar': 'Apply fog to each raw scan in its own sensor coordinates before loader transformations, sweep aggregation, voxelisation or camera-depth construction. Cover current keyframe and every historical sweep consumed. Preserve fifth raw feature; preserve loader calibration and time-lag construction.',
    'lidar_severity_alpha_beta': [[0.02, 0.008], [0.03, 0.008], [0.06, 0.05]],
    'lidar_source_options': 'simulate_fog(severity, raw_Nx5_float32, noise=10), hard=True, soft=True, gain=False, noise_variant=v1, output converted to float32 as in upstream converter.',
    'camera': 'All six decoded raw-resolution camera images, before existing augmentation/normalisation. Upstream image fog combines fractal haze with a spatial attenuation heuristic. Encoding and any acceleration must be validated and fixed in a subsequent implementation protocol before full inference.',
    'camera_beta_levels': [0.0005, 0.001, 0.002],
    'randomness_plan': 'One deterministic realisation per unique raw file and severity, shared across both pipelines. Freeze exact RNG mapping and any severity coupling in the implementation protocol after source-equivalence tests, before inference.',
    'pilot_rule': 'First token in the unchanged ordered 600-token list. Recover all six camera and all listed LiDAR inputs for that token; no model predictions or GT outcomes used. Never substitute another frame if data are missing.',
    'compatibility_gate': 'CPU checks on finite output, feature/channel preservation, source equivalence, repeatability, raw-scan processing and timing; then fresh clean/fog GPU smoke on preselected input. Full 600-frame runs only after acceptance. Record every implementation amendment before full inference.',
    'evaluation': 'Original detection_cvpr_2019 evaluator and unchanged GT filters on exact 600 tokens. mAP, NDS, per-class AP, errors and degradation from each fresh baseline. Descriptive results; scene/log correlation and one corruption realisation disclosed.',
    'limits': 'Synthetic benchmark corruption, not recorded real fog or a calibrated joint physical camera/LiDAR atmosphere. Severity indices do not establish a common visibility distance. No snow/disaster validation or controlled camera-component ablation.',
    'this_stage': 'Only source pinning, availability inventory and private pilot input transfer. No corruption, inference, accuracy metric, threshold selection or GPU environment mutation.'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def fetch_pinned(repo, ref, relative, expected, target):
    url = RAW + repo + '/' + ref + '/' + relative
    data = urllib.request.urlopen(url, timeout=120).read()
    git = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert git == expected, 'Pinned source mismatch: ' + relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data, {'url': url, 'git_blob_sha1': git, 'sha256': sha(data), 'bytes': len(data)}


def validate_population(token_data, inventory_data, sample_count=600, scene_count=150):
    tokens = token_data.decode().splitlines()
    assert len(tokens) == len(set(tokens)) == sample_count
    assert all(re.fullmatch('[0-9a-f]{32}', t) for t in tokens)
    inventory = json.loads(inventory_data)
    assert len(inventory) == sample_count
    by_token = {r['sample_token']: r for r in inventory}
    assert len(by_token) == sample_count and set(by_token) == set(tokens)
    scene_counts = Counter(r['scene_token'] for r in inventory)
    assert len(scene_counts) == scene_count and set(scene_counts.values()) == {4}
    required = set()
    channels = {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
    for row in inventory:
        cameras, lidar = row['camera_files'], row['lidar_files']
        assert len(cameras) == len(set(cameras)) == 6
        assert {PurePosixPath(n).parts[1] for n in cameras} == channels
        assert len(lidar) == len(set(lidar)) == 10
        for name in cameras + lidar:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] in ('samples', 'sweeps')
            required.add(name)
        # Some historical scans live under samples/: do not infer current scan
        # identity from the filename prefix, list order or number of such paths.
    return tokens, by_token, required


def sensor_index(names, required):
    found = {}
    for name in names:
        if name.endswith('/'):
            continue
        parts = PurePosixPath(name).parts
        if '..' in parts or name.startswith('/'):
            continue
        for i, part in enumerate(parts):
            if part not in ('samples', 'sweeps'):
                continue
            relative = '/'.join(parts[i:])
            if relative in required:
                assert relative not in found, 'Ambiguous sensor member: ' + relative
                found[relative] = name
                break
    return found


def candidates(base):
    """Bounded catalogue of the project folder, not the user's whole Drive."""
    files, folders = [], []
    allowed = re.compile(r'camera|challenge|availability|point.?reduction|frozen|600', re.I)
    queue = [(base, 0)]
    while queue:
        folder, depth = queue.pop(0)
        children = sorted(folder.iterdir(), key=lambda p: p.name)
        for child in children:
            if child.is_file() and child.suffix.lower() == '.zip':
                files.append(child)
            elif child.is_dir():
                if depth == 0: folders.append(child.name)
                if depth < 2 and allowed.search(child.name):
                    queue.append((child, depth + 1))
    files.sort(key=lambda p: (not bool(re.search(r'input|frozen|validation_parts|keyframes', p.name, re.I)), str(p)))
    return files, folders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation'))
    parser.add_argument('--work', type=Path, default=Path('/content/fog_preflight_20261029'))
    parser.add_argument('--sensor-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    assert args.base.is_dir(), 'Mount Drive first; preserved project folder must exist'
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.work / stamp
    dest = args.base / 'fog_20261029' / 'preflight' / stamp
    out.mkdir(parents=True, exist_ok=False); dest.mkdir(parents=True, exist_ok=False)
    write_json(out / 'study_design.json', PLAN)
    shutil.copyfile(out / 'study_design.json', dest / 'study_design.json')
    assert file_sha(out / 'study_design.json') == file_sha(dest / 'study_design.json')
    print('FOG DESIGN AND PILOT RULE PRESERVED BEFORE INPUT INSPECTION', flush=True)
    provenance, inputs = {}, {}
    for relative, expected in PROJECT_FILES.items():
        data, record = fetch_pinned('abjain182002/multimodal-perception', PROJECT_REF, relative,
                                    expected, out / 'frozen_population' / Path(relative).name)
        inputs[Path(relative).name] = data; provenance[relative] = record
    assert sha(inputs['sample_tokens.txt']) == PLAN['tokens_sha256']
    tokens, inventory, required = validate_population(inputs['sample_tokens.txt'], inputs['selected_input_manifest.json'])
    assert len(required) == 9600, 'Frozen unique sensor inventory differs from prior challenge'
    pilot = inventory[tokens[0]]
    pilot_files = set(pilot['camera_files'] + pilot['lidar_files'])
    write_json(out / 'pilot_selection.json', {'rule': PLAN['pilot_rule'], 'selected': pilot,
               'predictions_opened': False, 'annotations_or_outcomes_used_for_selection': False})
    for relative, expected in UPSTREAM_FILES.items():
        _, record = fetch_pinned('ika-rwth-aachen/MultiCorrupt', UPSTREAM_REF, relative,
                                 expected, out / 'upstream' / relative)
        provenance['MultiCorrupt/' + relative] = record
    write_json(out / 'pinned_source_audit.json', provenance)
    roots = args.sensor_root + [Path('/content/datasets/nuscenes_full'),
        Path('/kaggle/input/datasets/abhishekjain18/camera-challenges600-inputs-zip/dataset'),
        Path('/kaggle/working/camera_challenges600/dataset')]
    source_for, root_audit, archive_audit = {}, [], []
    for root in dict.fromkeys(roots):
        if not root.is_dir():
            root_audit.append({'path': str(root), 'exists': False}); continue
        available = [n for n in required if (root / n).is_file()]
        root_audit.append({'path': str(root), 'exists': True, 'required_files_found': len(available)})
        for name in available:
            source_for.setdefault(name, {'kind': 'file', 'path': str(root / name)})
    archives, folders = candidates(args.base)
    for archive in archives[:32]:
        print('INDEXING AVAILABLE INPUT ARCHIVE:', archive.name, flush=True)
        try:
            with zipfile.ZipFile(archive) as z:
                index = sensor_index(z.namelist(), required)
                archive_audit.append({'path': str(archive), 'bytes': archive.stat().st_size,
                    'required_files_found': len(index), 'members': len(z.infolist()),
                    'manifests_or_annotations': [n for n in z.namelist() if re.search(r'manifest|\.pkl$', n, re.I)][:30],
                    'nested_archives_not_expanded': [n for n in z.namelist() if n.endswith(('.tgz', '.tar.gz', '.tar'))][:20],
                    'whole_archive_hash_checked': False})
                for name, member in index.items():
                    source_for.setdefault(name, {'kind': 'zip_member', 'archive': str(archive), 'member': member})
        except (OSError, zipfile.BadZipFile, AssertionError) as exc:
            archive_audit.append({'path': str(archive), 'error': str(exc), 'used': False})
    pilot_records = {}
    for name in sorted(pilot_files & set(source_for)):
        source = source_for[name]
        if source['kind'] == 'file':
            data = Path(source['path']).read_bytes()
        else:
            with zipfile.ZipFile(source['archive']) as z:
                data = z.read(source['member'])  # ZIP CRC checked during read.
        assert data, name
        if name.endswith('.pcd.bin'): assert len(data) % 20 == 0, 'Expected float32 Nx5 raw scan'
        target = out / 'pilot_inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        pilot_records[name] = {**source, 'bytes': len(data), 'sha256': sha(data)}
    missing_pilot = sorted(pilot_files - set(pilot_records))
    packages = {}
    for name in ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d', 'numpy', 'scipy', 'opencv-python', 'opencv-python-headless'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    gpu = {'nvidia_smi_present': bool(shutil.which('nvidia-smi'))}
    if gpu['nvidia_smi_present']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=20)
            gpu.update({'returncode': result.returncode, 'output': result.stdout.strip()})
        except (OSError, subprocess.TimeoutExpired) as exc: gpu['error'] = str(exc)
    ready = not missing_pilot
    report = {'status': 'FOG_PREFLIGHT_PILOT_INPUTS_READY' if ready else 'FOG_PREFLIGHT_PILOT_INPUTS_INCOMPLETE',
        'samples': 600, 'scenes': 150, 'required_sensor_files': len(required),
        'available_required_files_by_name': len(source_for), 'pilot_token': tokens[0],
        'pilot_files_expected': len(pilot_files), 'pilot_files_recovered': len(pilot_records),
        'missing_pilot_files': missing_pilot, 'pilot_sources': pilot_records,
        'extracted_roots': root_audit, 'archive_index_audit': archive_audit,
        'candidate_archive_count': len(archives), 'archive_scan_limit': 32,
        'archives_not_scanned': [str(p) for p in archives[32:]], 'project_subfolders': folders,
        'packages': packages, 'gpu': gpu, 'python': platform.python_version(),
        'cpu_count': os.cpu_count(), 'local_disk_free_bytes': shutil.disk_usage(out).free,
        'script_sha256': file_sha(__file__), 'elapsed_seconds': round(time.monotonic() - started, 2),
        'scope': 'Pinned sources and filename availability; pilot bytes hashed, ZIP CRC checked where applicable. Full source archives and all 9600 input contents not hash-verified in this stage. No source pickle executed, no corruption/inference, no prediction contents read. Raw pilot inputs stay private.'}
    write_json(out / 'preflight_audit.json', report)
    shutil.copyfile(__file__, out / 'preflight_fog_colab.py')
    members = {str(f.relative_to(out)): {'bytes': f.stat().st_size, 'sha256': file_sha(f)}
               for f in sorted(out.rglob('*')) if f.is_file()}
    write_json(out / 'manifest.json', {'files': members})
    archive = out.parent / ('fog_preflight_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*members, 'manifest.json']:
            z.write(out / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in members.items():
            data = z.read(name); assert len(data) == record['bytes'] and sha(data) == record['sha256']
    target = dest / archive.name
    with archive.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    assert file_sha(target) == file_sha(archive)
    receipt = {'status': 'FOG_PREFLIGHT_AND_DRIVE_COPY_PASS', 'pilot_inputs_ready': ready,
               'archive': str(target), 'bytes': target.stat().st_size, 'sha256': file_sha(target),
               'required_sensor_files': len(required), 'available_required_files_by_name': len(source_for),
               'pilot_files_recovered': len(pilot_records), 'pilot_files_expected': len(pilot_files),
               'next': 'Upload the ZIP for source review, CPU fog compatibility tests and runtime/data restoration planning.'}
    write_json(dest / 'backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print('UPLOAD THIS ZIP:', target, flush=True)


if __name__ == '__main__':
    main()
