"""Reuse the preserved GPU smoke runtime; diagnose one-frame fusion repeats.

No setup, cache regeneration, upstream edit, fog inference or relaxed smoke gate.
Default and strict PyTorch determinism are separate diagnostic subprocesses.
"""
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
import zipfile

PRIOR = Path('/content/fog_gpu_smoke_20261029/20261029T080852328937Z')
BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation/fog_20261029')
PRIOR_ZIP_SHA = '5891509e3160a49f052bfed701a24ccc99e82bd6bd7150da9c14478438fbf189'
WORKER_SHA = 'acc79d1f2cc1433d1b44fdd6f99805b39891b9e7d965ec615682e2edfd8962b6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def compare_predictions(a, b):
    """Diagnostic permutation comparison with the ORIGINAL numeric tolerances."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    a = {k: np.asarray(v) for k, v in a.items()}
    b = {k: np.asarray(v) for k, v in b.items()}
    if any(a[k].shape != b[k].shape for k in a): return {'same_shapes': False}
    n = len(a['labels'])
    if n == 0: return {'same_shapes': True, 'original_positional_check_pass': True, 'same_prediction_set_with_original_tolerances': True, 'unmatched_under_original_tolerances': 0, 'matching_permutation': []}
    positional = bool(np.array_equal(a['labels'], b['labels']) and
        np.allclose(a['scores'], b['scores'], rtol=1e-5, atol=1e-5) and
        np.allclose(a['boxes'], b['boxes'], rtol=1e-5, atol=1e-4))
    valid = ((a['labels'][:, None] == b['labels'][None, :]) &
        np.isclose(a['scores'][:, None], b['scores'][None, :], rtol=1e-5, atol=1e-5) &
        np.isclose(a['boxes'][:, None, :], b['boxes'][None, :, :], rtol=1e-5, atol=1e-4).all(axis=2))
    rows, cols = linear_sum_assignment((~valid).astype(int))
    return {'same_shapes': True, 'original_positional_check_pass': positional,
        'positional_label_mismatches': int(np.count_nonzero(a['labels'] != b['labels'])),
        'same_prediction_set_with_original_tolerances': bool(valid[rows, cols].all()),
        'unmatched_under_original_tolerances': int(n - valid[rows, cols].sum()),
        'matching_permutation': cols.tolist(),
        'note': 'Diagnostic only. Does not replace the original smoke acceptance check.'}


def worker(args):
    import copy
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint, set_random_seed
    sys.path.insert(0, str(PRIOR / 'evidence/code'))
    import fog_gpu_smoke as original
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS
    runtime = Path('/content/fog_runtime'); repo = runtime / 'mmdetection3d'
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    assert sha(PRIOR / 'evidence/code/fog_gpu_smoke.py') == WORKER_SHA
    assert torch.__version__ == '2.1.2+cu121' and torch.cuda.device_count() == 1
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == original.SOURCE_COMMIT
    set_random_seed(20261029, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    strict = args.mode == 'deterministic'
    torch.use_deterministic_algorithms(strict, warn_only=False)
    cfg = Config.fromfile(str(PRIOR / 'evidence/predictions/lidar_camera_resolved_config.py'))
    cfg.model.img_backbone.init_cfg = None; cfg.load_from = None; cfg.resume = False
    name, expected = original.CHECKPOINTS['lidar_camera']; checkpoint = runtime / 'checkpoints' / name
    assert sha(checkpoint) == expected
    model = init_model(cfg, checkpoint=None, device='cuda:0')
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True); model.eval()
    ds_cfg = Config.fromfile(str(PRIOR / 'evidence/predictions/lidar_camera_level_0_dataset_config.py')).dataset
    import pickle
    source_bytes = (PRIOR / 'annotations_kaggle.pkl').read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == original.ANNOTATION_SHA
    source = pickle.loads(source_bytes)
    prepared = pickle.loads(Path(ds_cfg.ann_file).read_bytes())
    assert len(prepared['data_list']) == 1
    audit = json.loads((PRIOR / 'prepared/pilot_input_audit.json').read_text())
    expected_paths = {v['path'] for v in audit['conditions'][0]['source_files'].values()}
    assert {str(c[k]) for c, k in original.path_fields(prepared['data_list'][0])} == expected_paths
    restored = copy.deepcopy(prepared['data_list'][0])
    for (new, key), (old, old_key) in zip(original.path_fields(restored), original.path_fields(source['data_list'][0])):
        assert key == old_key; new[key] = old[old_key]
    original.same(restored, source['data_list'][0])
    original.same(prepared['metainfo'], source['metainfo'])
    dataset = DATASETS.build(ds_cfg); assert len(dataset) == 1
    assert dataset.get_data_info(0)['token'] == 'a7795dcadad44c1c85fa27eef18366f6'
    data = dataset[0]

    def fingerprint(value):
        if isinstance(value, torch.Tensor): value = value.detach().cpu().contiguous().numpy()
        if isinstance(value, np.ndarray):
            return {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, np.generic): return fingerprint(value.item())
        if isinstance(value, dict): return {str(k): fingerprint(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [fingerprint(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)): return value
        if isinstance(value, type): return {'class': value.__module__ + '.' + value.__qualname__}
        if hasattr(value, 'tensor'): return fingerprint(value.tensor)
        if hasattr(value, 'to_dict'): return fingerprint(value.to_dict())
        raise TypeError('Unrecognised packed input type: ' + str(type(value)))

    fingerprints = [fingerprint(data), fingerprint(dataset[0]), fingerprint(dataset[0])]
    write(out / 'input_reload_fingerprints.json', {'reloads_identical': fingerprints[0] == fingerprints[1] == fingerprints[2], 'reloads': fingerprints})
    first = {}; current = {}; hooks = []
    def capture(name, tensor):
        array = tensor.detach().cpu().contiguous().numpy().copy()
        ref = first.setdefault(name, array)
        current[name] = {'shape': list(array.shape), 'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'exact_to_first': bool(np.array_equal(array, ref)),
            'different_elements_to_first': int(np.count_nonzero(array != ref)),
            'max_abs_difference_to_first': float(np.max(np.abs(array.astype(np.float64) - ref.astype(np.float64)), initial=0))}
    def depth_hook(module, inputs): capture('projected_depth_before_dtransform', inputs[0])
    hooks.append(model.view_transform.dtransform.register_forward_pre_hook(depth_hook))
    def output_hook(name):
        def hook(module, inputs, output): capture(name, output)
        return hook
    for name in ('view_transform', 'pts_middle_encoder', 'fusion_layer'):
        hooks.append(getattr(model, name).register_forward_hook(output_hook(name)))
    predictions = []
    try:
        for index in range(3):
            current.clear(); packed = copy.deepcopy(data); before = fingerprint(packed)
            with torch.inference_mode():
                pred = model.test_step(pseudo_collate([packed]))[0].pred_instances_3d
            torch.cuda.synchronize()
            p = {'boxes': pred.bboxes_3d.tensor.detach().cpu().numpy().tolist(),
                 'scores': pred.scores_3d.detach().cpu().numpy().tolist(),
                 'labels': pred.labels_3d.detach().cpu().numpy().tolist()}
            assert np.isfinite(np.asarray(p['boxes'])).all() and np.isfinite(p['scores']).all()
            assert all(0 <= s <= 1 for s in p['scores']) and all(0 <= label < 10 for label in p['labels'])
            write(out / f'prediction_{index}.json', p); predictions.append(p)
            write(out / f'forward_{index}.json', {'intermediates': copy.deepcopy(current),
                'same_packed_input_before_forward': before == fingerprints[0],
                'original_cpu_input_unchanged': fingerprint(data) == fingerprints[0]})
            print('REPEAT DIAGNOSTIC FORWARD:', args.mode, index, len(p['labels']), flush=True)
        comparisons = [compare_predictions(predictions[0], p) for p in predictions[1:]]
        write(out / 'diagnostic.json', {'status': 'FUSION_BASELINE_REPEAT_DIAGNOSTIC_COMPLETE',
            'mode': args.mode, 'strict_pytorch_determinism': strict, 'gpu': torch.cuda.get_device_name(0),
            'checkpoint_sha256': expected, 'comparisons_to_first': comparisons,
            'scope': 'One fixed unperturbed frame. No fog-level inference, accuracy claim or acceptance-rule amendment.'})
    finally:
        for hook in hooks: hook.remove()


def launch():
    assert PRIOR.is_dir(), 'Original Colab session files missing; do not reinstall blindly.'
    assert BASE.is_dir(), 'Mount the project Drive first.'
    runtime = Path('/content/fog_runtime'); python = runtime / 'bevfusion310/bin/python'
    assert python.is_file(), 'Original isolated environment missing.'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path('/content/fog_repeat_diagnostic_20261029') / stamp
    out.mkdir(parents=True, exist_ok=False)
    dest = BASE / 'repeat_diagnostic' / stamp; dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    error = None
    try:
        prior_zip = BASE / 'gpu_smoke/20261029T080852328937Z/fog_gpu_smoke_20261029T080852328937Z.zip'
        assert sha(prior_zip) == PRIOR_ZIP_SHA
        manifest = json.loads((PRIOR / 'evidence/evidence_manifest.json').read_text())['files']
        with zipfile.ZipFile(prior_zip) as z:
            saved_manifest = z.read('evidence_manifest.json')
            assert saved_manifest == (PRIOR / 'evidence/evidence_manifest.json').read_bytes()
            assert set(z.namelist()) == set(manifest) | {'evidence_manifest.json'}
            for name, record in manifest.items():
                raw = z.read(name)
                assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
                assert sha(PRIOR / 'evidence' / name) == record['sha256']
        audit = json.loads((PRIOR / 'prepared/pilot_input_audit.json').read_text())
        assert (PRIOR / 'prepared/pilot_input_audit.json').read_bytes() == (PRIOR / 'evidence/pilot_input_audit.json').read_bytes()
        condition = audit['conditions'][0]; assert condition['level'] == 0
        for record in condition['source_files'].values():
            assert sha(record['path']) == record['sha256'] and Path(record['path']).stat().st_size == record['bytes']
        # Worker verifies baseline annotation fields against the exact original source.
        protocol = {'status': 'FUSION_REPEAT_DIAGNOSTIC_FIXED_BEFORE_EXECUTION',
            'prior_zip_sha256': PRIOR_ZIP_SHA, 'script_sha256': sha(__file__),
            'sample_token': audit['sample_token'], 'modes': ['default', 'deterministic'], 'forwards_per_mode': 3,
            'checks': 'Repeated dataset input fingerprints; cloned identical packed input; projected depth and branch feature differences; saved raw predictions; original positional and permutation-aware diagnostics with unchanged tolerances.',
            'strict_mode': 'PyTorch deterministic algorithms True, warn_only False; CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import. A failure is saved, not bypassed.',
            'scope': 'No model/source/cache changes, no reinstallation, no full fog inference and no amended smoke acceptance.'}
        write(out / 'protocol.json', protocol); shutil.copyfile(out / 'protocol.json', dest / 'protocol.json')
        assert sha(out / 'protocol.json') == sha(dest / 'protocol.json')
        for name in ('cuda_operations_audit.json', 'environment.txt'):
            shutil.copyfile(runtime / 'setup' / name, out / name)
        env = os.environ.copy(); repo = runtime / 'mmdetection3d'
        env.update({'CUDA_VISIBLE_DEVICES': '0', 'CUDA_HOME': str(runtime / 'cuda-12.1'),
            'MPLBACKEND': 'Agg', 'PYTHONPATH': str(repo) + ':' + str(PRIOR / 'evidence/code') + ':' + env.get('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': str(runtime / 'cuda-12.1/lib64') + ':' + env.get('LD_LIBRARY_PATH', ''),
            'PATH': str(runtime / 'cuda-12.1/bin') + ':' + env.get('PATH', ''), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        returns = {}
        for mode in ('default', 'deterministic'):
            folder = out / mode; folder.mkdir()
            command = [str(python), '-u', str(Path(__file__).resolve()), '--mode', mode, '--output', str(folder)]
            with (folder / 'run.log').open('x') as log:
                p = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in p.stdout: print(line, end='', flush=True); log.write(line); log.flush()
                    returns[mode] = p.wait()
                except BaseException:
                    p.terminate(); p.wait(); raise
        write(out / 'subprocess_results.json', {'return_codes': returns, 'smoke_pass_claimed': False})
    except BaseException:
        error = traceback.format_exc(); write(out / 'launcher_failure.json', {'traceback': error})
    files = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out / 'manifest.json', {'files': files})
    archive = out.parent / ('fog_repeat_diagnostic_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'manifest.json']: z.write(out / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in files.items():
            raw = z.read(name); assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    target = dest / archive.name
    with archive.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    assert sha(target) == sha(archive)
    receipt = {'status': 'FUSION_REPEAT_DIAGNOSTIC_EVIDENCE_PRESERVED', 'archive': str(target),
        'sha256': sha(target), 'bytes': target.stat().st_size, 'launcher_error': error is not None,
        'scope': 'Diagnostic only. Inspect mode return codes and reports; this status is not a smoke pass.'}
    write(dest / 'backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2)); print('UPLOAD THIS ZIP:', target)
    if error: raise RuntimeError('Diagnostic launcher stopped; evidence preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('default', 'deterministic'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode:
        assert args.output
        try: worker(args)
        except BaseException:
            write(args.output / 'failure.json', {'traceback': traceback.format_exc()}); raise
    else: launch()
