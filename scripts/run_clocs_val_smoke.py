"""Score four frozen KITTI validation frames with PointPillars, YOLO26x, CLOCs.

Exports pre-NMS candidates and fusion scores. Detection AP is not calculated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import extract_dense_pointpillars_candidates, load_pointpillars_model
from fusion.clocs import build_clocs_feature_rows
from fusion.clocs_network import CLOCsFusionNet
from scripts.build_clocs_yolo_train_frame import car_detections
from scripts.package_clocs_val_smoke import FRAME_IDS, EXTENSIONS


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pointpillars-checkpoint', type=Path, required=True)
    parser.add_argument('--yolo-weights', type=Path, required=True)
    parser.add_argument('--fusion-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (manifest['schema_version'] != 'clocs_validation_smoke_inputs_v1'
            or manifest['frame_ids'] != list(FRAME_IDS)
            or sha256(args.pointpillars_checkpoint) != manifest['pointpillars_checkpoint_sha256']
            or sha256(args.fusion_checkpoint) != manifest['fusion_checkpoint_sha256']
            or sha256(args.yolo_weights) != '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
            or sha256(args.config) != '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc'):
        raise ValueError('One or more frozen model/config/input fingerprints differ')
    for fid in FRAME_IDS:
        for part, suffix in EXTENSIONS.items():
            path = args.dataset / part / (fid + suffix)
            if sha256(path) != manifest['input_sha256'][fid][part]:
                raise ValueError(f'Validation input differs from package: {path}')

    import torch
    from PIL import Image
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('PointPillars export requires CUDA')
    device = torch.device('cuda:0')
    payload = torch.load(args.fusion_checkpoint, map_location=device, weights_only=False)
    if (payload['epoch'] != 12 or payload['protocol']['settings']['negative_ratio'] != 0
            or not set(payload['split']['train_frame_ids']).isdisjoint(FRAME_IDS)
            or not set(payload['split']['dev_frame_ids']).isdisjoint(FRAME_IDS)):
        raise ValueError('Fusion checkpoint contains unexpected epoch, sampling, or split')
    fusion = CLOCsFusionNet().to(device).eval()
    fusion.load_state_dict(payload['model_state_dict'])
    pp = load_pointpillars_model(args.config, args.pointpillars_checkpoint, device='cuda:0')
    camera = YOLO(str(args.yolo_weights))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for fid in FRAME_IDS:
        target = args.output / f'{fid}_validation_scores.npz'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite validation result: {target}')
        image = args.dataset / 'image_2' / f'{fid}.png'
        with Image.open(image) as handle:
            width, height = handle.size
        prediction = camera.predict(source=str(image), device='cuda:0', conf=0.25, verbose=False)[0]
        boxes_2d, scores_2d = car_detections(prediction, camera.names)
        dense = extract_dense_pointpillars_candidates(pp,
            args.dataset / 'velodyne' / f'{fid}.bin',
            args.dataset / 'calib' / f'{fid}.txt', width, height)
        rows = build_clocs_feature_rows(dense['projected_boxes'], dense['scores_3d'],
                                        boxes_2d, scores_2d, dense['distance_82'])
        with torch.no_grad():
            logits = fusion(torch.from_numpy(rows['features']).to(device),
                            torch.from_numpy(rows['candidate_index'].astype(np.int64)).to(device),
                            dense['num_candidates'])['candidate_logits']
            fused = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        if fused.shape != dense['scores_3d'].shape or not np.isfinite(fused).all():
            raise ValueError(f'Invalid fused score vector for {fid}')
        np.savez_compressed(target, frame_id=fid, split='validation',
            boxes_lidar=dense['boxes_lidar'], boxes_camera=dense['boxes_camera'],
            scores_3d=dense['scores_3d'], fused_scores=fused,
            direction_labels=dense['direction_labels'],
            projected_boxes=dense['projected_boxes'],
            detector_2d_boxes=boxes_2d, detector_2d_scores=scores_2d,
            pointpillars_checkpoint_sha256=manifest['pointpillars_checkpoint_sha256'],
            fusion_checkpoint_sha256=manifest['fusion_checkpoint_sha256'],
            input_manifest_sha256=sha256(args.manifest))
        item = dict(frame_id=fid, candidates=dense['num_candidates'],
                    yolo26x_car_detections=int(len(scores_2d)),
                    num_feature_rows=int(len(rows['features'])),
                    score_file_sha256=sha256(target))
        summary.append(item)
        print(json.dumps(item), flush=True)
    (args.output / 'inference_audit.json').write_text(json.dumps(dict(
        schema_version='clocs_val_smoke_inference_v1', status='PASS',
        interpretation='Four validation frames scored before detection postprocessing; no KITTI AP measured.',
        manifest_sha256=sha256(args.manifest),
        fusion_checkpoint_sha256=manifest['fusion_checkpoint_sha256'],
        pointpillars_checkpoint_sha256=manifest['pointpillars_checkpoint_sha256'],
        frames=summary), indent=2) + '\n')


if __name__ == '__main__':
    main()
