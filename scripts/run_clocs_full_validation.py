"""Stream frozen PointPillars and YOLO26x+CLOCs validation detections.

One compact JSON file is committed per frame. Reruns verify and reuse completed
frames. Evaluation is a separate operation after all 3769 frames finish.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from adapters.pointpillars_adapter import extract_dense_pointpillars_candidates, load_pointpillars_model
from fusion.clocs import build_clocs_feature_rows
from fusion.clocs_network import CLOCsFusionNet
from fusion.kitti_reduced_cloud import reduce_kitti_frame
from scripts.build_clocs_yolo_train_frame import car_detections
from scripts.evaluate_clocs_val_subset import (detection_annotation, gt_annotation,
                                                load_kitti_calibration, postprocess)

EXPECTED = {
    'pointpillars': 'd42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e',
    'fusion': 'a2203b9adee46b782e896b7f3b9344b5916433b5404d4bd871aa7e05d6b0c473',
    'yolo': '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92',
    'config': '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc',
    'train': 'b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb',
    'val': '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest, allow_partial=False):
    partial = manifest.get('schema_version') == 'clocs_kitti_validation_batch_v1'
    if (manifest.get('schema_version') not in (
            ('clocs_kitti_full_validation_inputs_v1', 'clocs_kitti_validation_batch_v1')
            if allow_partial else ('clocs_kitti_full_validation_inputs_v1',))
            or manifest.get('split') != 'validation'
            or manifest.get('train_split_sha256') != EXPECTED['train']
            or manifest.get('val_split_sha256') != EXPECTED['val']):
        raise ValueError('Full validation manifest differs from the frozen inputs')
    ids = manifest.get('frame_ids', [])
    if ((not partial and (len(ids) != 3769 or manifest.get('input_files') != 15076))
            or (partial and (not ids or manifest.get('input_files') != 4 * len(ids)
                             or manifest.get('parent_manifest_sha256') is None))
            or len(set(ids)) != len(ids) or not all(
            isinstance(fid, str) and len(fid) == 6 and fid.isdigit() for fid in ids)):
        raise ValueError('Expected 3769 unique validation IDs')
    return ids


def provenance_for(manifest, manifest_path):
    parent = manifest.get('parent_manifest_sha256')
    return dict(input_manifest_sha256=parent or sha256(manifest_path), **EXPECTED)


def to_json(annotation):
    return {key: value.tolist() for key, value in annotation.items()}


def verified_existing(path, fid, provenance):
    if not path.exists():
        return False
    item = json.loads(path.read_text())
    if (item.get('frame_id') != fid or item.get('provenance') != provenance
            or set(item.get('detections', {})) != {'pointpillars', 'fusion'}
            or 'ground_truth' not in item):
        raise ValueError(f'Existing frame output requires inspection: {path}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pointpillars-checkpoint', type=Path, required=True)
    parser.add_argument('--yolo-weights', type=Path, required=True)
    parser.add_argument('--fusion-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--point-cloud-mode', choices=('raw', 'reduced_fov'),
                        default='raw')
    parser.add_argument('--historical-predictions', type=Path,
                        help='Verified historical pickle for per-frame baseline parity')
    parser.add_argument('--max-new-frames', type=int, default=0,
                        help='0 processes all missing frames; small values allow a smoke run')
    args = parser.parse_args()

    import torch
    from PIL import Image
    from ultralytics import YOLO
    from mmengine.config import Config
    from importlib.metadata import version

    if version('ultralytics') != '8.4.132':
        raise ValueError('Use the frozen Ultralytics 8.4.132 camera inference')

    manifest = json.loads(args.manifest.read_text())
    ids = verify_manifest(manifest, allow_partial=True)
    if (sha256(args.config) != EXPECTED['config']
            or sha256(args.pointpillars_checkpoint) != EXPECTED['pointpillars']
            or sha256(args.yolo_weights) != EXPECTED['yolo']
            or sha256(args.fusion_checkpoint) != EXPECTED['fusion']):
        raise ValueError('A frozen checkpoint, detector, or configuration differs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for dense PointPillars and fusion inference')
    cfg = Config.fromfile(str(args.config)).model.test_cfg
    if (cfg.nms_pre, cfg.nms_thr, cfg.score_thr, cfg.max_num,
            cfg.use_rotate_nms) != (100, .01, .1, 50, True):
        raise ValueError('Pinned postprocessing settings differ')

    device = torch.device('cuda:0')
    payload = torch.load(args.fusion_checkpoint, map_location=device, weights_only=False)
    trained = set(payload['split']['train_frame_ids']) | set(payload['split']['dev_frame_ids'])
    if (payload['epoch'] != 12
            or payload['protocol']['settings']['negative_ratio'] != 0
            or trained.intersection(ids)):
        raise ValueError('Fusion checkpoint split or training protocol differs')
    fusion = CLOCsFusionNet().to(device).eval()
    fusion.load_state_dict(payload['model_state_dict'])
    pp = load_pointpillars_model(args.config, args.pointpillars_checkpoint, device='cuda:0')
    camera = YOLO(str(args.yolo_weights))

    args.output.mkdir(parents=True, exist_ok=True)
    provenance = provenance_for(manifest, args.manifest)
    historical = None
    if args.point_cloud_mode == 'reduced_fov':
        provenance['point_cloud_mode'] = 'reduced_fov'
        provenance['reduction'] = ('MMDetection3D fe25f7a5 KITTI converter '
                                   'camera-2 field-of-view')
    if args.historical_predictions:
        import pickle
        expected_history = ('2905f9c15fd04a6055dca99753b6fb6e6585492371608c5a625e3cc6ded764d3')
        if sha256(args.historical_predictions) != expected_history:
            raise ValueError('Historical predictions SHA-256 mismatch; refusing to unpickle')
        with args.historical_predictions.open('rb') as stream:
            historical = pickle.load(stream)
        if not isinstance(historical, list) or len(historical) != len(ids):
            raise ValueError('Historical predictions must contain 3769 ordered frames')
    created = reused = 0
    started = time.monotonic()
    for index, fid in enumerate(ids, 1):
        target = args.output / f'{fid}.json'
        if verified_existing(target, fid, provenance):
            reused += 1
            continue
        if args.max_new_frames and created >= args.max_new_frames:
            break
        files = {folder: args.dataset / folder / (fid + ext)
                 for folder, ext in (('velodyne', '.bin'), ('image_2', '.png'),
                                     ('calib', '.txt'), ('label_2', '.txt'))}
        if not all(path.is_file() for path in files.values()):
            raise FileNotFoundError(f'Missing validation input for {fid}: {files}')
        with Image.open(files['image_2']) as image:
            width, height = image.size
        prediction = camera.predict(source=str(files['image_2']), device='cuda:0',
                                    conf=.25, verbose=False)[0]
        boxes_2d, scores_2d = car_detections(prediction, camera.names)
        calib = load_kitti_calibration(files['calib'])
        cloud = files['velodyne']
        reduced_cloud = None
        if args.point_cloud_mode == 'reduced_fov':
            reduced_cloud = args.output / '.reduced_clouds' / f'{fid}.bin'
            if reduced_cloud.exists():
                raise FileExistsError(f'Partial reduced cloud requires inspection: {reduced_cloud}')
            reduce_kitti_frame(cloud, calib, height, width, reduced_cloud)
            cloud = reduced_cloud
        try:
            dense = extract_dense_pointpillars_candidates(
                pp, cloud, files['calib'], width, height)
        finally:
            if reduced_cloud is not None:
                reduced_cloud.unlink()
        rows = build_clocs_feature_rows(dense['projected_boxes'], dense['scores_3d'],
                                        boxes_2d, scores_2d, dense['distance_82'])
        with torch.inference_mode():
            logits = fusion(torch.from_numpy(rows['features']).to(device),
                            torch.from_numpy(rows['candidate_index'].astype(np.int64)).to(device),
                            dense['num_candidates'])['candidate_logits']
            fused = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        if fused.shape != dense['scores_3d'].shape or not np.isfinite(fused).all():
            raise ValueError(f'Invalid fusion score vector for {fid}')
        detected = {}
        for name, score in (('pointpillars', dense['scores_3d']), ('fusion', fused)):
            kept, confidence, labels = postprocess(
                dense['boxes_lidar'], dense['direction_labels'], score, cfg)
            detected[name] = to_json(detection_annotation(
                kept, confidence, labels, calib, width, height))
        if historical is not None:
            old = historical[index - 1]
            now = detected['pointpillars']
            samples = np.asarray(old['sample_idx'])
            if (len(samples) != len(old['score']) or np.any(samples != index - 1)
                    or len(old['score']) != len(now['score'])):
                raise ValueError(f'Historical baseline count/index mismatch: {fid}')
            if len(old['score']):
                for field, tolerance in (('score', 1e-5), ('bbox', 1e-3),
                                         ('dimensions', 1e-4), ('location', 1e-4),
                                         ('rotation_y', 1e-5)):
                    if not np.allclose(np.asarray(old[field]), np.asarray(now[field]),
                                       rtol=0, atol=tolerance):
                        raise ValueError(f'Historical baseline {field} mismatch: {fid}')
        item = dict(frame_id=fid, provenance=provenance,
                    ground_truth=to_json(gt_annotation(files['label_2'])),
                    detections=detected,
                    candidate_count=int(dense['num_candidates']),
                    camera_detections=int(len(scores_2d)))
        temporary = args.output / f'.{fid}.partial.json'
        if temporary.exists():
            raise FileExistsError(f'Partial frame output requires inspection: {temporary}')
        temporary.write_text(json.dumps(item, separators=(',', ':')) + '\n')
        os.replace(temporary, target)
        created += 1
        print(json.dumps(dict(frame=fid, index=index, total=len(ids),
                              created=created, reused=reused,
                              baseline=len(detected['pointpillars']['score']),
                              fusion=len(detected['fusion']['score']),
                              elapsed_seconds=round(time.monotonic() - started, 1))),
              flush=True)
    print(json.dumps(dict(status='COMPLETE' if created + reused == len(ids) else 'PARTIAL',
                          created=created, reused=reused, expected=len(ids),
                          output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
