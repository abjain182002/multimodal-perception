"""Frozen 600-frame fog cache generation and strict deterministic inference.

One condition is prepared once and consumed by both separately pretrained models.
No GT accuracy evaluation or training. NuScenes JSON uses pinned upstream formatter.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

PRIOR = Path('/content/fog_gpu_smoke_20261029/20261029T080852328937Z')
RAW_ROOT = Path('/content/fog_full_inputs_20261029/dataset')
ORIGINAL_WORKER_SHA = 'acc79d1f2cc1433d1b44fdd6f99805b39891b9e7d965ec615682e2edfd8962b6'
ADAPTER_SHA = '495a2027eb108cda21ac736a460b478c844f5064e216cab9e9b27b9832693be2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def helpers():
    folder = PRIOR / 'evidence/code'
    assert sha(folder / 'fog_gpu_smoke.py') == ORIGINAL_WORKER_SHA
    assert sha(folder / 'fog_transforms.py') == ADAPTER_SHA
    assert sha(folder / 'preflight_fog_colab.py') == 'c919c0f36579d6026315a302909593076b69a19e6b64274039a0228fb9cd5bf7'
    sys.path.insert(0, str(folder))
    import fog_gpu_smoke
    return fog_gpu_smoke


def source_annotation(original, required):
    data = (PRIOR / 'annotations_kaggle.pkl').read_bytes()
    assert hashlib.sha256(data).hexdigest() == original.ANNOTATION_SHA
    source = pickle.loads(data)  # Exact own preserved annotation SHA checked first.
    tokens = (PRIOR / 'pilot/frozen_population/sample_tokens.txt').read_text().splitlines()
    assert len(source['data_list']) == len(tokens) == 600
    assert [row['token'] for row in source['data_list']] == tokens
    for row in source['data_list']:
        assert len(row['images']) == 6 and len(row['lidar_sweeps']) == 9
        for container, key in original.path_fields(row): original.relative_path(container[key], required)
    return source, tokens


def rewrite_annotation(source, outputs, original):
    result = copy.deepcopy(source)
    allowed = set(outputs)
    for new_row, old_row in zip(result['data_list'], source['data_list']):
        for (new, key), (old, old_key) in zip(original.path_fields(new_row), original.path_fields(old_row)):
            assert key == old_key
            relative = original.relative_path(old[old_key], allowed)
            new[key] = outputs[relative]['path']
        restored = copy.deepcopy(new_row)
        for (new, key), (old, old_key) in zip(original.path_fields(restored), original.path_fields(old_row)):
            assert key == old_key; new[key] = old[old_key]
        original.same(restored, old_row)
    original.same(result['metainfo'], source['metainfo'])
    return result


_ADAPTER = None

def init_adapter():
    global _ADAPTER
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import cv2
    cv2.setNumThreads(1)
    helpers()
    from fog_transforms import FogAdapter
    _ADAPTER = FogAdapter(PRIOR / 'pilot/upstream')


def generate(task):
    import cv2
    import numpy as np
    relative, record, level, cache = task
    root = Path(cache) / 'dataset'
    target = root / (relative + '.png' if relative.endswith('.jpg') else relative)
    sidecar = Path(cache) / 'checks' / (relative + '.json')
    source = RAW_ROOT / relative
    assert source.stat().st_size == record['bytes'] and sha(source) == record['sha256']
    identity = {'source_sha256': record['sha256'], 'level': level, 'adapter_sha256': ADAPTER_SHA, 'relative_path': relative}
    if sidecar.is_file() and target.is_file():
        saved = json.loads(sidecar.read_text())
        assert saved['identity'] == identity and sha(target) == saved['sha256']
        return relative, saved
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + '.partial' + target.suffix)
    if relative.endswith('.jpg'):
        image = cv2.imread(str(source)); assert image is not None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = _ADAPTER.camera(rgb, level, relative)
        assert cv2.imwrite(str(temporary), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        restored = cv2.cvtColor(cv2.imread(str(temporary)), cv2.COLOR_BGR2RGB)
        assert np.array_equal(output, restored)
        extra = {'image_shape': list(output.shape), 'lossless_pixel_roundtrip': True}
    else:
        points = np.fromfile(source, dtype=np.float32).reshape(-1, 5)
        output, stats = _ADAPTER.points(points, level, relative)
        output.tofile(temporary)
        assert np.array_equal(np.fromfile(temporary, dtype=np.float32).reshape(-1, 5), output)
        extra = {'raw_points': len(output), **stats}
    temporary.replace(target)
    saved = {'identity': identity, 'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target), **extra}
    atomic(sidecar, saved)
    return relative, saved


def prepare(args):
    original = helpers()
    restored = json.loads(args.restoration.read_text())
    assert restored['status'] == 'FOG_FULL_600_FRAME_SENSOR_CONTENT_RESTORATION_PASS'
    required = restored['members']; assert len(required) == 9600
    source, tokens = source_annotation(original, set(required))
    assert shutil.disk_usage(args.cache.parent).free > 22 * 1024**3, 'Need >22 GiB free for one lossless condition and output headroom'
    outputs = {}; started = time.monotonic()
    tasks = [(relative, record, args.level, str(args.cache)) for relative, record in sorted(required.items())]
    # Separate spawned processes preserve the adapter's legacy camera RNG isolation.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=init_adapter) as pool:
        for index, (relative, saved) in enumerate(pool.map(generate, tasks, chunksize=1), 1):
            outputs[relative] = saved
            if index % 100 == 0 or index == len(tasks):
                print(f'FOG CACHE level={args.level}: {index}/9600 files; {time.monotonic()-started:.1f}s', flush=True)
    assert set(outputs) == set(required)
    annotation = rewrite_annotation(source, outputs, original)
    ann_path = args.cache / 'infos.pkl'
    temp = ann_path.with_suffix('.partial'); temp.write_bytes(pickle.dumps(annotation, protocol=4)); temp.replace(ann_path)
    # Compare full post-serialization geometry/metadata as well.
    original.same(pickle.loads(ann_path.read_bytes()), annotation)
    pilot = json.loads((PRIOR / 'prepared/pilot_input_audit.json').read_text())['conditions'][args.level]['source_files']
    for relative, saved in pilot.items(): assert outputs[relative]['sha256'] == saved['sha256'], 'Pilot cache differs from accepted smoke'
    report = {'status': 'FOG_FULL_CONDITION_CACHE_AND_ANNOTATION_PASS', 'level': args.level,
        'samples': len(tokens), 'files': outputs, 'annotation': str(ann_path), 'annotation_sha256': sha(ann_path),
        'original_annotation_sha256': original.ANNOTATION_SHA, 'non_path_metadata_unchanged': True,
        'accepted_pilot_cache_hashes_match': True, 'elapsed_seconds': round(time.monotonic()-started, 2),
        'dataset_root': str(args.cache / 'dataset'), 'scope': 'Raw input corruption/cache preparation only; all 9600 inputs and 600 annotation rows retained.'}
    report_path = args.output / 'cache_audit.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        assert {k:v for k,v in saved.items() if k != 'elapsed_seconds'} == {k:v for k,v in report.items() if k != 'elapsed_seconds'}, 'Existing condition identity differs'
    else: atomic(report_path, report)


def predict(args):
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint, set_random_seed
    original = helpers()
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.evaluation.metrics import NuScenesMetric
    runtime = Path('/content/fog_runtime'); repo = runtime / 'mmdetection3d'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert torch.__version__ == '2.1.2+cu121' and torch.cuda.device_count() == 1
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == original.SOURCE_COMMIT
    assert subprocess.run(['git', 'diff', '--quiet'], cwd=repo).returncode == 0, 'Tracked upstream source has local changes'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_num_threads(1)
    set_random_seed(20261029, deterministic=False)
    cfg = Config.fromfile(str(PRIOR / 'evidence/predictions' / f'{args.model}_resolved_config.py'))
    cfg.load_from = None; cfg.resume = False
    if cfg.model.get('img_backbone') is not None: cfg.model.img_backbone.init_cfg = None
    checkpoint_name, checkpoint_sha = original.CHECKPOINTS[args.model]
    checkpoint = runtime / 'checkpoints' / checkpoint_name; assert sha(checkpoint) == checkpoint_sha
    cache = json.loads((args.output.parent / 'cache_audit.json').read_text())
    assert cache['level'] == args.level and sha(cache['annotation']) == cache['annotation_sha256']
    cfg.dump(str(args.output / 'resolved_config.py'))
    model = init_model(cfg, checkpoint=None, device='cuda:0')
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True); model.eval()
    dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset_cfg.data_root = cache['dataset_root']; dataset_cfg.ann_file = cache['annotation']
    dataset_cfg.test_mode = True
    dataset_cfg.data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP', img='')
    Config(dict(dataset=dataset_cfg)).dump(str(args.output / 'dataset_config.py'))
    dataset = DATASETS.build(dataset_cfg)
    tokens = (PRIOR / 'pilot/frozen_population/sample_tokens.txt').read_text().splitlines()
    assert len(dataset) == len(tokens) == 600
    assert [dataset.get_data_info(i)['token'] for i in range(600)] == tokens
    infos = pickle.loads(Path(cache['annotation']).read_bytes())['data_list']
    all_required = set(cache['files']); assert len(all_required) == 9600
    # Check every parsed path, not only filenames recorded in the cache manifest.
    parsed = set()
    for index in range(600):
        info = dataset.get_data_info(index)
        for container, key in original.path_fields(info):
            path = str(container[key]); assert Path(path).is_file(); parsed.add(path)
    assert parsed == {r['path'] for r in cache['files'].values()}
    identity = {'model': args.model, 'level': args.level, 'worker_sha256': sha(__file__),
        'cache_audit_sha256': sha(args.output.parent / 'cache_audit.json'), 'checkpoint_sha256': checkpoint_sha,
        'annotation_sha256': cache['annotation_sha256'], 'source_commit': original.SOURCE_COMMIT}
    records = []; formatted = []; created = reused = 0; started = time.monotonic()
    raw = args.output / 'frames'; raw.mkdir(exist_ok=True)
    smoke_root = Path(args.smoke)
    for index, token in enumerate(tokens):
        sample = dataset[index]; points = sample['inputs']['points']
        point_sha = hashlib.sha256(points.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        record_path = raw / f'{index:04d}_{token}.json'
        if record_path.is_file():
            saved = json.loads(record_path.read_text())
            assert saved['identity'] == identity and saved['sample_token'] == token and saved['packed_points_sha256'] == point_sha
            reused += 1
        else:
            with torch.inference_mode(): pred = model.test_step(pseudo_collate([sample]))[0].pred_instances_3d
            torch.cuda.synchronize()
            saved = {'identity': identity, 'sample_token': token, 'index': index,
                'packed_points_sha256': point_sha, 'packed_points_shape': list(points.shape),
                'boxes_lidar': pred.bboxes_3d.tensor.detach().cpu().numpy().tolist(),
                'scores': pred.scores_3d.detach().cpu().numpy().tolist(), 'labels': pred.labels_3d.detach().cpu().numpy().tolist()}
            atomic(record_path, saved); assert json.loads(record_path.read_text()) == saved
            created += 1
        boxes = np.asarray(saved['boxes_lidar'], dtype=np.float32).reshape(-1, 9)
        scores = np.asarray(saved['scores'], dtype=np.float32); labels = np.asarray(saved['labels'], dtype=np.int64)
        assert len(boxes) == len(scores) == len(labels)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all() and ((labels >= 0) & (labels < 10)).all()
        if index == 0:
            accepted = json.loads((smoke_root / f'predictions/{args.model}_level_{args.level}_predictions.json').read_text())
            assert point_sha == json.loads((smoke_root / f'predictions/{args.model}_smoke_audit.json').read_text())['conditions'][args.level]['packed_points_sha256']
            np.testing.assert_array_equal(labels, np.asarray(accepted['labels']))
            np.testing.assert_allclose(scores, np.asarray(accepted['scores']), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(boxes, np.asarray(accepted['boxes_lidar']), rtol=1e-5, atol=1e-4)
        formatted.append({'sample_idx': index, 'pred_instances_3d': {'bboxes_3d': LiDARInstance3DBoxes(torch.from_numpy(boxes), box_dim=9),
            'scores_3d': torch.from_numpy(scores), 'labels_3d': torch.from_numpy(labels)}})
        records.append({'index': index, 'sample_token': token, 'packed_points_sha256': point_sha,
            'prediction_sha256': sha(record_path), 'prediction_count': len(boxes)})
        if (index+1) % 25 == 0 or index == 0:
            print(f'INFERENCE level={args.level} model={args.model}: {index+1}/600; new={created} reused={reused}; {time.monotonic()-started:.1f}s', flush=True)
    metric = NuScenesMetric(data_root=cache['dataset_root'], ann_file=cache['annotation'],
        modality={'use_lidar': True, 'use_camera': args.model == 'lidar_camera'},
        format_only=True, jsonfile_prefix=str(args.output / 'formatted'))
    metric.dataset_meta = dataset.metainfo; metric.data_infos = infos; metric.version = 'v1.0-trainval'
    paths, _ = metric.format_results(formatted, classes=dataset.metainfo['classes'], jsonfile_prefix=str(args.output / 'formatted'))
    shutil.copyfile(paths['pred_instances_3d'], args.output / 'results_nusc.json')
    assert sha(paths['pred_instances_3d']) == sha(args.output / 'results_nusc.json')
    shutil.rmtree(args.output / 'formatted')
    output = json.loads((args.output / 'results_nusc.json').read_text())
    assert set(output['results']) == set(tokens) and len(output['results']) == 600
    assert all(all(x['sample_token'] == token for x in values) for token, values in output['results'].items())
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    atomic(args.output / 'input_checks.json', records)
    atomic(args.output / 'inference_audit.json', {'status': 'FOG_FULL_MODEL_CONDITION_INFERENCE_PASS', **identity,
        'samples': 600, 'exact_frozen_tokens': True, 'accepted_pilot_prediction_match': True,
        'strict_determinism': True, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'results_sha256': sha(args.output / 'results_nusc.json'), 'input_checks_sha256': sha(args.output / 'input_checks.json'),
        'created': created, 'reused': reused, 'elapsed_seconds': round(time.monotonic()-started, 2),
        'scope': 'Inference/export only. No AP/NDS, training or full-study accuracy conclusion.'})
    print('FOG_FULL_MODEL_CONDITION_INFERENCE_PASS', args.model, args.level, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','predict'))
    parser.add_argument('--level', type=int, choices=(0,1,2,3), required=True)
    parser.add_argument('--cache', type=Path); parser.add_argument('--restoration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('lidar','lidar_camera')); parser.add_argument('--smoke', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare': prepare(args)
    else: predict(args)


if __name__ == '__main__': main()
