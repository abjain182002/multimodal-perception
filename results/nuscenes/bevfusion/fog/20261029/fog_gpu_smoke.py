"""One fixed frame, two pinned models, unperturbed + three fog levels.

Runs only in the isolated project GPU environment. No GT accuracy evaluation.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess

import cv2
import numpy as np

from fog_transforms import FogAdapter

CHECKPOINTS = {
    'lidar': ('bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth',
              '2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e'),
    'lidar_camera': ('bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth',
                     'e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab')}
ANNOTATION_SHA = '4f11db818b8d5504fc2bb432fcd51a28b09fec8d0b9c4063f2a017413fbfa99d'
SOURCE_COMMIT = 'fe25f7a51d36e3702f961e198894580d83c4387b'


def sha(data): return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def relative_path(value, allowed):
    parts = Path(str(value)).parts
    matches = ['/'.join(parts[i:]) for i, p in enumerate(parts) if p in ('samples', 'sweeps') and '/'.join(parts[i:]) in allowed]
    if not matches:
        matches = [p for p in allowed if Path(p).name == str(value)]
    assert len(matches) == 1, ('Ambiguous/unrecognised annotation path', value)
    return matches[0]


def same(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left: same(left[k], right[k])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    elif isinstance(left, (float, np.floating)):
        assert left == right or (np.isnan(left) and np.isnan(right))
    else: assert left == right


def path_fields(row):
    yield row['lidar_points'], 'lidar_path'
    for record in row['images'].values(): yield record, 'img_path'
    for record in row.get('lidar_sweeps', []): yield record['lidar_points'], 'lidar_path'


def prepare(args):
    selected = json.loads((args.pilot / 'pilot_selection.json').read_text())['selected']
    adapter = FogAdapter(args.pilot / 'upstream')
    allowed = set(selected['camera_files'] + selected['lidar_files'])
    data = args.annotation.read_bytes(); assert sha(data) == ANNOTATION_SHA
    original = pickle.loads(data)  # Own preserved project annotation; exact known SHA checked first.
    tokens = (args.pilot / 'frozen_population/sample_tokens.txt').read_text().splitlines()
    assert len(original['data_list']) == len(tokens) == 600
    assert [r['token'] for r in original['data_list']] == tokens
    assert selected['sample_token'] == tokens[0]
    row = original['data_list'][0]
    assert len(row['images']) == 6 and len(row.get('lidar_sweeps', [])) == 9
    assert {relative_path(c[k], allowed) for c, k in path_fields(row)} == allowed
    records = []
    for level in (0, 1, 2, 3):
        root = args.prepared / f'level_{level}' / 'dataset'
        outputs = {}
        for relative in sorted(allowed):
            source = args.pilot / 'pilot_inputs' / relative
            name = relative + '.png' if relative in selected['camera_files'] else relative
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            if relative in selected['camera_files']:
                rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
                output = adapter.camera(rgb, level, relative)
                assert cv2.imwrite(str(target), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
                decoded = cv2.cvtColor(cv2.imread(str(target)), cv2.COLOR_BGR2RGB)
                assert np.array_equal(output, decoded)
            else:
                points = np.fromfile(source, dtype=np.float32).reshape(-1, 5)
                output, _ = adapter.points(points, level, relative)
                output.tofile(target)
                assert np.array_equal(np.fromfile(target, dtype=np.float32).reshape(-1, 5), output)
            outputs[relative] = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target.read_bytes())}
        one = {'metainfo': copy.deepcopy(original['metainfo']), 'data_list': [copy.deepcopy(row)]}
        for container, key in path_fields(one['data_list'][0]):
            relative = relative_path(container[key], allowed)
            container[key] = outputs[relative]['path']
        restored = copy.deepcopy(one['data_list'][0])
        for (new, key), (old, old_key) in zip(path_fields(restored), path_fields(row)):
            assert key == old_key; new[key] = old[old_key]
        same(restored, row); same(one['metainfo'], original['metainfo'])
        annotation = root.parent / 'pilot_infos.pkl'
        annotation.write_bytes(pickle.dumps(one, protocol=4))
        records.append({'level': level, 'annotation': str(annotation), 'data_root': str(root),
                        'source_files': outputs, 'non_path_row_fields_unchanged': True})
    args.prepared.mkdir(parents=True, exist_ok=True)
    write_json(args.prepared / 'pilot_input_audit.json', {'status': 'FOG_PILOT_CACHES_AND_ANNOTATIONS_PASS',
        'sample_token': tokens[0], 'conditions': records, 'annotation_source_sha256': ANNOTATION_SHA,
        'inference_performed': False, 'scope': 'One fixed frame. Six cameras and all nine historical sweeps plus current raw scan per level. PNG encoding preserves decoded pixels; only annotation paths and smoke subset membership change.'})


def predict(args):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import load_checkpoint, set_random_seed
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS

    assert torch.__version__ == '2.1.2+cu121' and torch.cuda.device_count() == 1
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip() == SOURCE_COMMIT
    set_random_seed(20261029, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    names = {'lidar': 'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
             'lidar_camera': 'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'}
    cfg = Config.fromfile(str(args.repo / 'projects/BEVFusion/configs' / names[args.model]))
    if cfg.model.get('img_backbone') is not None: cfg.model.img_backbone.init_cfg = None
    cfg.load_from = None; cfg.resume = False
    filename, expected = CHECKPOINTS[args.model]
    checkpoint = args.checkpoints / filename
    assert sha(checkpoint.read_bytes()) == expected
    cfg.dump(str(args.output / f'{args.model}_resolved_config.py'))
    model = init_model(cfg, checkpoint=None, device='cuda:0')
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model.eval()
    audit = json.loads((args.prepared / 'pilot_input_audit.json').read_text())
    results = []
    for condition in audit['conditions']:
        dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
        dataset_cfg.data_root = condition['data_root']
        dataset_cfg.ann_file = condition['annotation']
        dataset_cfg.test_mode = True
        dataset_cfg.data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP', img='')
        Config(dict(dataset=dataset_cfg)).dump(str(args.output / f'{args.model}_level_{condition["level"]}_dataset_config.py'))
        dataset = DATASETS.build(dataset_cfg)
        assert len(dataset) == 1 and dataset.get_data_info(0)['token'] == audit['sample_token']
        data = dataset[0]
        points = data['inputs']['points']
        point_hash = sha(points.detach().cpu().contiguous().numpy().tobytes())
        with torch.inference_mode():
            prediction = model.test_step(pseudo_collate([data]))[0].pred_instances_3d
        torch.cuda.synchronize()
        boxes = prediction.bboxes_3d.tensor.detach().cpu().numpy()
        scores = prediction.scores_3d.detach().cpu().numpy()
        labels = prediction.labels_3d.detach().cpu().numpy()
        assert len(boxes) == len(scores) == len(labels)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all() and ((labels >= 0) & (labels < 10)).all()
        if condition['level'] == 0:
            with torch.inference_mode():
                repeat = model.test_step(pseudo_collate([dataset[0]]))[0].pred_instances_3d
            np.testing.assert_array_equal(labels, repeat.labels_3d.detach().cpu().numpy())
            np.testing.assert_allclose(scores, repeat.scores_3d.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(boxes, repeat.bboxes_3d.tensor.detach().cpu().numpy(), rtol=1e-5, atol=1e-4)
        level = condition['level']
        write_json(args.output / f'{args.model}_level_{level}_predictions.json',
                   {'sample_token': audit['sample_token'], 'boxes_lidar': boxes.tolist(),
                    'scores': scores.tolist(), 'labels': labels.tolist()})
        results.append({'level': level, 'prediction_count': len(boxes), 'packed_points': list(points.shape),
                        'packed_points_sha256': point_hash, 'finite_predictions': True})
        print('FOG GPU FORWARD PASS:', args.model, level, len(boxes), flush=True)
    write_json(args.output / f'{args.model}_smoke_audit.json', {'status': 'FOG_FOUR_CONDITION_GPU_SMOKE_PASS',
        'model': args.model, 'sample_token': audit['sample_token'], 'checkpoint_sha256': expected,
        'gpu': torch.cuda.get_device_name(0), 'conditions': results, 'clean_repeat_pass': True,
        'source_commit': SOURCE_COMMIT, 'scope': 'One predetermined frame; strict checkpoint load, valid finite predictions and clean repeat. No AP/NDS, runtime benchmark or robustness claim.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--pilot', type=Path)
    p.add_argument('--annotation', type=Path)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--repo', type=Path)
    p.add_argument('--checkpoints', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--model', choices=('lidar', 'lidar_camera'))
    args = p.parse_args()
    if args.prepare:
        assert args.pilot and args.annotation; prepare(args)
    else:
        assert args.repo and args.checkpoints and args.output and args.model
        args.output.mkdir(parents=True, exist_ok=True); predict(args)


if __name__ == '__main__': main()
