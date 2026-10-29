"""Colab GPU smoke launcher: one frozen frame, not the full 600-frame study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

REPO = 'abjain182002/multimodal-perception'
FOLDER = 'results/nuscenes/bevfusion/fog/20261029/'
PREFLIGHT_SHA = '9fd0f95619b8e4d19ef9a4c6432e332a90d75281c3a303159e76581bdb6a8126'
ANNOTATION_SHA = '4f11db818b8d5504fc2bb432fcd51a28b09fec8d0b9c4063f2a017413fbfa99d'
SOURCE_SHA256 = {
    'preflight_fog_colab.py': 'c919c0f36579d6026315a302909593076b69a19e6b64274039a0228fb9cd5bf7',
    'fog_transforms.py': '495a2027eb108cda21ac736a460b478c844f5064e216cab9e9b27b9832693be2',
    'fog_gpu_smoke.py': 'acc79d1f2cc1433d1b44fdd6f99805b39891b9e7d965ec615682e2edfd8962b6',
    'setup_fog_gpu.py': '670a0d63f407a4bbd0efd7ca1b236e12c22204f9628f9d711c06140f90d6be3f',
    'cpu_validation/cpu_validation.json': '8d9dfcf68814c3d8dc4883802172e02735041b3070fc2ff9cd9ccc13a97a9ad7'}
CHECKPOINTS = {
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth': '2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e',
    'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth': 'e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def run(command, log, cwd=None, env=None):
    print('RUNNING:', ' '.join(map(str, command)), flush=True)
    with log.open('x') as f:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
    assert code == 0, f'Command failed ({code}); log: {log}'


def download(url, target, expected=None):
    if target.exists():
        assert expected and file_sha(target) == expected; return
    temp = target.with_name(target.name + '.partial')
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    if expected: assert file_sha(temp) == expected, 'Downloaded hash mismatch'
    temp.replace(target)


def unpack_pilot(archive, root):
    assert file_sha(archive) == PREFLIGHT_SHA
    root.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('manifest.json'))['files']
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        for name, record in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data = z.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)


def archive_evidence(out, dest, stamp, status):
    files = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
             for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'evidence_manifest.json', {'files': files})
    source = out.parent / ('fog_gpu_smoke_' + stamp + '.zip')
    with zipfile.ZipFile(source, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'evidence_manifest.json']: z.write(out / name, name)
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        for name, record in files.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    target = dest / source.name
    with source.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    assert file_sha(source) == file_sha(target)
    receipt = {'status': status, 'archive': str(target), 'bytes': target.stat().st_size,
               'sha256': file_sha(target), 'payload_members': len(files),
               'scope': 'One fixed frame. GPU setup/smoke evidence, not full-study accuracy.'}
    write_json(dest / 'backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print('UPLOAD THIS ZIP:', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation'))
    args = parser.parse_args()
    assert args.base.is_dir(), 'Mount Drive first'
    assert len(args.code_commit) == 40 and all(c in '0123456789abcdef' for c in args.code_commit)
    assert shutil.which('nvidia-smi'), 'Select a GPU runtime before running this cell; T4 is the previously used GPU.'
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    assert gpu.returncode == 0, gpu.stdout + gpu.stderr
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent = Path('/content/fog_gpu_smoke_20261029') / stamp
    out = parent / 'evidence'; out.mkdir(parents=True, exist_ok=False)
    code = out / 'code'; code.mkdir()
    dest = args.base / 'fog_20261029' / 'gpu_smoke' / stamp
    dest.mkdir(parents=True, exist_ok=False)
    (out / 'initial_nvidia_smi.txt').write_text(gpu.stdout)
    shutil.copyfile(__file__, code / 'run_fog_gpu_smoke_colab.py')
    error = None
    try:
        downloads = {}
        for name, expected in SOURCE_SHA256.items():
            target = code / Path(name).name
            download(f'https://raw.githubusercontent.com/{REPO}/{args.code_commit}/{FOLDER}{name}', target, expected)
            data = target.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            downloads[name] = {'git_blob_sha1': blob, 'sha256': file_sha(target)}
        cpu = json.loads((code / 'cpu_validation.json').read_text())
        assert cpu['status'] == 'FOG_REAL_PILOT_CPU_SOURCE_EQUIVALENCE_PASS'
        assert cpu['adapter_sha256'] == file_sha(code / 'fog_transforms.py')
        protocol = {'status': 'FOG_GPU_SMOKE_IMPLEMENTATION_FIXED_BEFORE_INFERENCE',
            'code_commit': args.code_commit, 'code_files': downloads,
            'pilot_token': cpu['sample_token'], 'preflight_zip_sha256': PREFLIGHT_SHA,
            'conditions': ['unperturbed', 'fog_1', 'fog_2', 'fog_3'], 'models': ['lidar', 'lidar_camera'],
            'randomness': cpu['method_seed'], 'encoding': cpu['encoding_amendment'],
            'arithmetic': cpu['arithmetic_amendment'],
            'controls': 'Pinned detector commit/checkpoints; batch size 1, eval mode, no autocast, TF32 off, cudnn benchmark off. All six cameras and all nine raw sweeps plus keyframe. Same point inputs required across pipelines per condition.',
            'acceptance': 'CUDA NMS/pooling/voxel tests; strict checkpoint loads; valid finite scores, labels and boxes for every condition; clean-repeat labels exact, scores rtol/atol 1e-5, boxes rtol 1e-5 atol 1e-4. No performance-based frame substitution.',
            'scope': 'One-frame implementation smoke only; no accuracy, runtime benchmark or full-study fog robustness claim.'}
        write_json(out / 'implementation_protocol.json', protocol)
        shutil.copyfile(out / 'implementation_protocol.json', dest / 'implementation_protocol.json')
        assert file_sha(out / 'implementation_protocol.json') == file_sha(dest / 'implementation_protocol.json')
        preflight = args.base / 'fog_20261029/preflight/20261029T072247008844Z/fog_preflight_20261029T072247008844Z.zip'
        pilot = parent / 'pilot'; unpack_pilot(preflight, pilot)
        shutil.copyfile(pilot / 'pinned_source_audit.json', out / 'pinned_upstream_source_audit.json')
        # This first pilot comes entirely from the original preserved source.
        # Re-anchor its bytes to the known full-archive hash before inference.
        original_archive = args.base / 'nuscenes_validation_parts_01_02_verified.zip'
        print('VERIFYING ORIGINAL PILOT SOURCE ARCHIVE BEFORE INFERENCE', flush=True)
        original_hash = file_sha(original_archive)
        assert original_hash == 'c89bc342a240d0bde22ef8b9200ebaf6de5b5f91c8a399073c4057d866f39c3a'
        pilot_sources = json.loads((pilot / 'preflight_audit.json').read_text())['pilot_sources']
        with zipfile.ZipFile(original_archive) as z:
            for relative, record in pilot_sources.items():
                assert record['kind'] == 'zip_member' and Path(record['archive']).name == original_archive.name
                data = z.read(record['member'])
                assert hashlib.sha256(data).hexdigest() == record['sha256']
                assert file_sha(pilot / 'pilot_inputs' / relative) == record['sha256']
        write_json(out / 'pilot_original_source_verification.json', {'archive': original_archive.name,
            'sha256': original_hash, 'fresh_full_archive_rehash': True, 'pilot_members_rechecked': len(pilot_sources)})
        annotation = parent / 'annotations_kaggle.pkl'
        for archive in [args.base / 'camera_challenges600_20261016/camera_challenges600_inference_export.zip',
                        args.base / 'camera_challenges600_inference_export.zip']:
            if archive.is_file():
                with zipfile.ZipFile(archive) as z:
                    if 'study/annotations_kaggle.pkl' not in z.namelist(): continue
                    data = z.read('study/annotations_kaggle.pkl')
                    assert hashlib.sha256(data).hexdigest() == ANNOTATION_SHA
                    annotation.write_bytes(data); break
        assert annotation.is_file(), 'Preserved 600-frame annotation export missing'
        write_json(out / 'annotation_source.json', {'archive': str(archive), 'member': 'study/annotations_kaggle.pkl',
                    'sha256': ANNOTATION_SHA, 'whole_export_archive_rehashed': False})
        print('SETTING UP ISOLATED GPU STACK; downloads/builds precede the short smoke.', flush=True)
        run([sys.executable, '-u', code / 'setup_fog_gpu.py'], out / 'gpu_setup.log')
        runtime = Path('/content/fog_runtime')
        python = runtime / 'bevfusion310/bin/python'
        repo = runtime / 'mmdetection3d'
        env = os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES': '0', 'CUDA_HOME': str(runtime / 'cuda-12.1'),
            'CC': '/usr/bin/gcc-11', 'CXX': '/usr/bin/g++-11', 'MAX_JOBS': '2', 'MPLBACKEND': 'Agg',
            'PYTHONPATH': str(repo) + ':' + str(code) + ':' + env.get('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': str(runtime / 'cuda-12.1/lib64') + ':' + env.get('LD_LIBRARY_PATH', ''),
            'PATH': str(runtime / 'cuda-12.1/bin') + ':' + env.get('PATH', '')})
        checkpoints = runtime / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
        for name, expected in CHECKPOINTS.items():
            download('https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/' + name,
                     checkpoints / name, expected)
        prepared = parent / 'prepared'
        worker = code / 'fog_gpu_smoke.py'
        run([python, '-u', worker, '--prepare', '--pilot', pilot, '--annotation', annotation,
             '--prepared', prepared], out / 'pilot_preparation.log', cwd=repo, env=env)
        shutil.copyfile(prepared / 'pilot_input_audit.json', out / 'pilot_input_audit.json')
        for model in ('lidar', 'lidar_camera'):
            run([python, '-u', worker, '--prepared', prepared, '--repo', repo,
                 '--checkpoints', checkpoints, '--output', out / 'predictions', '--model', model],
                out / (model + '_smoke.log'), cwd=repo, env=env)
        a = json.loads((out / 'predictions/lidar_smoke_audit.json').read_text())
        b = json.loads((out / 'predictions/lidar_camera_smoke_audit.json').read_text())
        assert a['sample_token'] == b['sample_token'] == cpu['sample_token']
        for left, right in zip(a['conditions'], b['conditions']):
            assert left['level'] == right['level']
            assert left['packed_points_sha256'] == right['packed_points_sha256'], 'Packed LiDAR inputs differ between pipelines'
        write_json(out / 'paired_smoke_audit.json', {'status': 'PAIRED_FOG_GPU_SMOKE_PASS',
            'sample_token': a['sample_token'], 'conditions_per_model': 4, 'fresh_clean_repeat_each_model': True,
            'same_packed_point_tensors_per_condition': True, 'full_600_frame_inference_performed': False,
            'next': 'Review the smoke evidence, then restore/verify all 9600 inputs and launch bounded full-study runs.'})
        for name in ('cuda_operations_audit.json', 'environment.txt'):
            shutil.copyfile(runtime / 'setup' / name, out / name)
    except BaseException as exc:
        error = exc
        write_json(out / 'failure.json', {'status': 'FOG_GPU_SMOKE_FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
    status = 'FOG_GPU_SMOKE_AND_DRIVE_COPY_PASS' if error is None else 'FOG_GPU_SMOKE_FAILED_EVIDENCE_PRESERVED'
    archive_evidence(out, dest, stamp, status)
    if error is not None: raise RuntimeError('GPU smoke stopped; failure evidence ZIP preserved above.') from error


if __name__ == '__main__': main()
