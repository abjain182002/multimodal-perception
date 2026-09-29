"""Evaluate complete streamed Car detections with the pinned KITTI AP40 logic.

Full fusion comparison is blocked unless PointPillars reproduces the established
3769-frame baseline. Use NUMBA_CUDA_USE_NVIDIA_BINDING=1 with cuda-python 12.2.1.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.run_clocs_full_validation import EXPECTED, sha256, verify_manifest

BASELINE = {
    '2D': (96.10463590066347, 92.4841064377867, 89.76978062674723),
    'BEV': (92.76631753879718, 88.36802380600489, 85.50537785461054),
    '3D': (90.34490342609234, 79.12987417920098, 76.15655595797031),
}


def array_annotation(source, detection):
    item = {key: np.asarray(value) for key, value in source.items()}
    item['bbox'] = np.asarray(source['bbox'], dtype=np.float64).reshape(-1, 4)
    item['dimensions'] = np.asarray(source['dimensions'], dtype=np.float64).reshape(-1, 3)
    item['location'] = np.asarray(source['location'], dtype=np.float64).reshape(-1, 3)
    for key in ('bbox', 'dimensions', 'location'):
        if len(item[key]) != len(item['name']):
            raise ValueError(f'Invalid annotation length: {key}')
    if detection and (len(item['score']) != len(item['name'])
                      or any(name != 'Car' for name in item['name'])):
        raise ValueError('Invalid Car detection annotation')
    return item


def evaluate(gt, dt):
    from mmdet3d.evaluation.functional.kitti_utils.eval import eval_class, get_mAP40

    overlap = np.asarray([[[.7], [.7], [.7]]], dtype=np.float64)
    result = {}
    for name, metric in (('2D', 0), ('BEV', 1), ('3D', 2)):
        precision = eval_class(gt, dt, [0], [0, 1, 2], metric, overlap,
                               compute_aos=False, num_parts=50)['precision']
        values = get_mAP40(precision)[0, :, 0]
        result[name] = dict(zip(('easy', 'moderate', 'hard'), map(float, values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--point-cloud-mode', choices=('raw', 'reduced_fov'),
                        default='raw')
    args = parser.parse_args()

    if os.environ.get('NUMBA_CUDA_USE_NVIDIA_BINDING') != '1' or os.environ.get('NUMBA_ENABLE_CUDASIM'):
        raise RuntimeError('Use the pinned NVIDIA binding, without the Numba CUDA simulator')
    from importlib.metadata import version
    if version('cuda-python') != '12.2.1':
        raise RuntimeError('The full evaluator requires cuda-python 12.2.1')
    ids = verify_manifest(json.loads(args.manifest.read_text()))
    provenance = dict(input_manifest_sha256=sha256(args.manifest), **EXPECTED)
    if args.point_cloud_mode == 'reduced_fov':
        provenance['point_cloud_mode'] = 'reduced_fov'
        provenance['reduction'] = ('MMDetection3D fe25f7a5 KITTI converter '
                                   'camera-2 field-of-view')
    gt = []
    predictions = {'pointpillars': [], 'fusion': []}
    hashes = hashlib.sha256()
    counts = {'pointpillars': 0, 'fusion': 0}
    for fid in ids:
        path = args.predictions / f'{fid}.json'
        raw = path.read_bytes()
        hashes.update(bytes.fromhex(hashlib.sha256(raw).hexdigest()))
        frame = json.loads(raw)
        if frame.get('frame_id') != fid or frame.get('provenance') != provenance:
            raise ValueError(f'Incorrect prediction or provenance for frame {fid}')
        gt.append(array_annotation(frame['ground_truth'], detection=False))
        for name in predictions:
            annotation = array_annotation(frame['detections'][name], detection=True)
            predictions[name].append(annotation)
            counts[name] += len(annotation['score'])
    print(f'INPUTS PASS: {len(ids)} ordered frames; retained detections {counts}', flush=True)

    baseline = evaluate(gt, predictions['pointpillars'])
    differences = {name: {difficulty: baseline[name][difficulty] - reference
                          for difficulty, reference in zip(('easy', 'moderate', 'hard'), values)}
                   for name, values in BASELINE.items()}
    largest = max(abs(value) for metric in differences.values() for value in metric.values())
    report = dict(schema_version='clocs_full_val_ap40_v1',
                  status='BASELINE_PARITY_PENDING' if largest > .05 else 'PASS',
                  interpretation=('Frozen 3769-frame KITTI SECOND validation, Car strict IoU 0.70. '
                                  'Fusion comparison accepted only after baseline AP parity.'),
                  frames=len(ids), point_cloud_mode=args.point_cloud_mode,
                  evaluator='MMDetection3D pinned fe25f7a5',
                  overlap_backend='Numba NVIDIA CUDA binding; cuda-python 12.2.1',
                  source_sha256=dict(input_manifest=sha256(args.manifest),
                                     ordered_prediction_digests=hashes.hexdigest()),
                  detections=counts, baseline_max_absolute_ap40_error=largest,
                  baseline_ap40_differences=differences,
                  ap40_strict_iou_070={'pointpillars': baseline})
    if largest <= .05:
        report['ap40_strict_iou_070']['fusion'] = evaluate(gt, predictions['fusion'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if largest > .05:
        raise RuntimeError(f'Baseline differs from established full validation by {largest:.4f} AP points')


if __name__ == '__main__':
    main()
