"""Resume the fixed fog pilot under documented strict PyTorch determinism.

Reuses the installed runtime and all four prepared conditions; no setup/downloads.
Original worker source and numeric acceptance tolerances are retained verbatim.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

PRIOR = Path('/content/fog_gpu_smoke_20261029/20261029T080852328937Z')
BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation/fog_20261029')
ORIGINAL_ZIP_SHA = '5891509e3160a49f052bfed701a24ccc99e82bd6bd7150da9c14478438fbf189'
DIAGNOSTIC_ZIP_SHA = '4ef30b9a808cabd729edc9ada7116a800ac52b7832a94def2d7ce6d4f51565b2'
WORKER_SHA = 'acc79d1f2cc1433d1b44fdd6f99805b39891b9e7d965ec615682e2edfd8962b6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, data):
    with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def check_zip(path, expected, manifest_name):
    assert sha(path) == expected
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read(manifest_name))['files']
        assert set(z.namelist()) == set(manifest) | {manifest_name}
        for name, rec in manifest.items():
            data = z.read(name)
            assert len(data) == rec['bytes'] and hashlib.sha256(data).hexdigest() == rec['sha256']
        return manifest


def child(model, out):
    import copy
    import pickle
    import runpy
    import torch
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    # The verified original worker never disables this global flag. Its seed helper's
    # deterministic=False does not change torch.use_deterministic_algorithms.
    torch.backends.cudnn.deterministic = True
    worker = PRIOR / 'evidence/code/fog_gpu_smoke.py'
    assert sha(worker) == WORKER_SHA
    sys.path.insert(0, str(worker.parent))
    import fog_gpu_smoke as original
    source_bytes = (PRIOR / 'annotations_kaggle.pkl').read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == original.ANNOTATION_SHA
    source = pickle.loads(source_bytes)
    audit = json.loads((PRIOR / 'prepared/pilot_input_audit.json').read_text())
    assert audit['sample_token'] == 'a7795dcadad44c1c85fa27eef18366f6'
    assert [c['level'] for c in audit['conditions']] == [0, 1, 2, 3]
    for condition in audit['conditions']:
        cached = pickle.loads(Path(condition['annotation']).read_bytes())
        assert len(cached['data_list']) == 1
        paths = {v['path'] for v in condition['source_files'].values()}
        assert {str(c[k]) for c, k in original.path_fields(cached['data_list'][0])} == paths
        restored = copy.deepcopy(cached['data_list'][0])
        for (new, key), (old, old_key) in zip(original.path_fields(restored), original.path_fields(source['data_list'][0])):
            assert key == old_key; new[key] = old[old_key]
        original.same(restored, source['data_list'][0]); original.same(cached['metainfo'], source['metainfo'])
    runtime = Path('/content/fog_runtime')
    previous = sys.argv[:]
    try:
        sys.argv = [str(worker), '--prepared', str(PRIOR / 'prepared'), '--repo', str(runtime / 'mmdetection3d'),
                    '--checkpoints', str(runtime / 'checkpoints'), '--output', str(out), '--model', model]
        runpy.run_path(str(worker), run_name='__main__')
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        write(out / (model + '_determinism_audit.json'), {
            'strict_deterministic_algorithms_enabled_after_all_forwards': True,
            'warn_only': False, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'original_worker_sha256': WORKER_SHA, 'source_model_code_changed': False,
            'numeric_acceptance_tolerances_changed': False})
    finally:
        sys.argv = previous


def main():
    assert BASE.is_dir() and PRIOR.is_dir(), 'Keep the original Colab runtime and mounted project Drive.'
    runtime = Path('/content/fog_runtime'); python = runtime / 'bevfusion310/bin/python'
    assert python.is_file(), 'Original isolated runtime is missing; do not reinstall blindly.'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path('/content/fog_gpu_smoke_deterministic_20261029') / stamp / 'evidence'
    out.mkdir(parents=True, exist_ok=False)
    dest = BASE / 'gpu_smoke_deterministic' / stamp; dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    error = None
    try:
        old_zip = BASE / 'gpu_smoke/20261029T080852328937Z/fog_gpu_smoke_20261029T080852328937Z.zip'
        files = check_zip(old_zip, ORIGINAL_ZIP_SHA, 'evidence_manifest.json')
        for name, rec in files.items(): assert sha(PRIOR / 'evidence' / name) == rec['sha256']
        diagnostic = BASE / 'repeat_diagnostic/20261029T084930614337Z/fog_repeat_diagnostic_20261029T084930614337Z.zip'
        check_zip(diagnostic, DIAGNOSTIC_ZIP_SHA, 'manifest.json')
        with zipfile.ZipFile(diagnostic) as z:
            result = json.loads(z.read('deterministic/diagnostic.json'))
            assert all(c['original_positional_check_pass'] for c in result['comparisons_to_first'])
        audit_path = PRIOR / 'prepared/pilot_input_audit.json'
        assert audit_path.read_bytes() == (PRIOR / 'evidence/pilot_input_audit.json').read_bytes()
        audit = json.loads(audit_path.read_text()); checked = 0
        assert [c['level'] for c in audit['conditions']] == [0, 1, 2, 3]
        for condition in audit['conditions']:
            assert len(condition['source_files']) == 16
            for rec in condition['source_files'].values():
                assert Path(rec['path']).stat().st_size == rec['bytes'] and sha(rec['path']) == rec['sha256']; checked += 1
        protocol = {'status': 'FOG_STRICT_DETERMINISM_AMENDMENT_PRESERVED_BEFORE_RESMOKE',
            'script_sha256': sha(__file__), 'original_worker_sha256': WORKER_SHA,
            'diagnostic_zip_sha256': DIAGNOSTIC_ZIP_SHA, 'original_failed_zip_sha256': ORIGINAL_ZIP_SHA,
            'sample_token': audit['sample_token'], 'cached_sensor_files_rechecked': checked,
            'amendment': 'Both models: strict PyTorch deterministic algorithms True (warn_only False), cudnn deterministic True, CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import. Original worker retains TF32 off, benchmark off, seed 20261029, strict checkpoints and unchanged positional repeat tolerances.',
            'reason': 'Identical packed inputs gave varying projected depth and outputs under default execution; three strict deterministic baseline forwards were bitwise identical on this T4 pilot.',
            'limits': 'Does not prove cross-device/version repeatability or isolate a single CUDA kernel. Does not change the physical meaning of the synthetic fog.',
            'controls': 'Same frozen frame, caches, checkpoints, source commit, calibration and numeric acceptance. All four conditions rerun for both models; only baseline is repeated, as in the original gate.',
            'historical_comparison': 'Fresh strict-deterministic baseline required for every full-study comparison. Do not overwrite historical default-execution results or pool their predictions with new results.',
            'full_600_frame_inference_performed': False}
        write(out / 'implementation_amendment.json', protocol)
        shutil.copyfile(out / 'implementation_amendment.json', dest / 'implementation_amendment.json')
        assert sha(out / 'implementation_amendment.json') == sha(dest / 'implementation_amendment.json')
        shutil.copyfile(audit_path, out / 'pilot_input_audit.json')
        shutil.copyfile(PRIOR / 'evidence/code/fog_gpu_smoke.py', out / 'original_fog_gpu_smoke.py')
        operations = json.loads((runtime / 'setup/cuda_operations_audit.json').read_text())
        for extension in operations['extensions'].values(): assert sha(extension['path']) == extension['sha256']
        for name in ('cuda_operations_audit.json', 'environment.txt'): shutil.copyfile(runtime / 'setup' / name, out / name)
        env = os.environ.copy(); repo = runtime / 'mmdetection3d'
        env.update({'CUDA_VISIBLE_DEVICES': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'CUDA_HOME': str(runtime / 'cuda-12.1'), 'MPLBACKEND': 'Agg',
            'PYTHONPATH': str(repo) + ':' + str(PRIOR / 'evidence/code') + ':' + env.get('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': str(runtime / 'cuda-12.1/lib64') + ':' + env.get('LD_LIBRARY_PATH', ''),
            'PATH': str(runtime / 'cuda-12.1/bin') + ':' + env.get('PATH', '')})
        predictions = out / 'predictions'; predictions.mkdir()
        for model in ('lidar', 'lidar_camera'):
            cmd = [str(python), '-u', str(Path(__file__).resolve()), '--worker', model, str(predictions)]
            with (out / (model + '.log')).open('x') as log:
                process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout: print(line, end='', flush=True); log.write(line); log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate(); process.wait(); raise
            assert code == 0, f'{model} stopped; inspect preserved log'
        a, b = [json.loads((predictions / (model + '_smoke_audit.json')).read_text()) for model in ('lidar', 'lidar_camera')]
        assert a['sample_token'] == b['sample_token'] == audit['sample_token']
        for model_audit in (a, b):
            assert model_audit['clean_repeat_pass'] and [c['level'] for c in model_audit['conditions']] == [0, 1, 2, 3]
        for left, right in zip(a['conditions'], b['conditions']):
            assert left['level'] == right['level'] and left['packed_points_sha256'] == right['packed_points_sha256']
        write(out / 'paired_smoke_audit.json', {'status': 'PAIRED_STRICT_DETERMINISTIC_FOG_GPU_SMOKE_PASS',
            'sample_token': audit['sample_token'], 'conditions_per_model': 4,
            'original_positional_baseline_repeat_pass_each_model': True,
            'same_packed_points_per_condition': True, 'strict_determinism_each_model': True,
            'full_600_frame_inference_performed': False, 'scope': 'One-frame implementation gate only. No AP/NDS or full-study accuracy claim.'})
    except BaseException:
        error = traceback.format_exc(); write(out / 'failure.json', {'traceback': error})
    records = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out / 'manifest.json', {'files': records})
    archive = out.parent / ('fog_gpu_smoke_deterministic_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*records, 'manifest.json']: z.write(out / name, name)
    check_zip(archive, sha(archive), 'manifest.json')
    target = dest / archive.name
    with archive.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    assert sha(target) == sha(archive)
    receipt = {'status': 'FOG_STRICT_SMOKE_AND_DRIVE_COPY_PASS' if error is None else 'FOG_STRICT_SMOKE_FAILED_EVIDENCE_PRESERVED',
        'archive': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target), 'scope': 'One fixed frame, not full-study accuracy.'}
    write(dest / 'backup_receipt.json', receipt); print(json.dumps(receipt, indent=2)); print('UPLOAD THIS ZIP:', target)
    if error: raise RuntimeError('Strict smoke stopped; failure evidence preserved.')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        output = Path(sys.argv[3])
        try: child(sys.argv[2], output)
        except BaseException:
            write(output / (sys.argv[2] + '_failure.json'), {'traceback': traceback.format_exc()}); raise
    else:
        assert len(sys.argv) == 1; main()
