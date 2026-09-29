"""Evaluate frozen four-frame validation scores with pinned KITTI AP40 code.

Subset AP40 is an engineering check and must never be reported as full KITTI AP.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration

FRAMES = ('000831', '001662', '003324', '004155')
CONFIG_SHA = '75145966859d3f9f4bc3fcd364ee12e6b3edc221cf1cdea7c70e4274443891dc'
POINTPILLARS_SHA = 'd42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e'
FUSION_SHA = 'a2203b9adee46b782e896b7f3b9344b5916433b5404d4bd871aa7e05d6b0c473'


def postprocess(boxes, directions, scores, cfg):
    import torch
    from mmdet3d.models.layers import box3d_multiclass_nms
    from mmdet3d.structures import LiDARInstance3DBoxes, limit_period, xywhr2xyxyr

    box_tensor = torch.as_tensor(boxes, device='cuda:0')
    dirs = torch.as_tensor(directions, device='cuda:0')
    values = torch.as_tensor(scores, device='cuda:0')
    if (box_tensor.shape != (107136, 7) or dirs.shape != (107136,)
            or values.shape != (107136,) or not torch.isfinite(values).all()):
        raise ValueError('Invalid saved candidate shape or scores')
    top = values.topk(cfg.nms_pre).indices
    chosen_boxes, chosen_scores = box_tensor[top], values[top]
    bev = LiDARInstance3DBoxes(chosen_boxes, box_dim=7).bev
    nms_boxes = xywhr2xyxyr(bev)
    classes = torch.stack((chosen_scores, torch.zeros_like(chosen_scores)), dim=1)
    kept, final_scores, labels, kept_dirs = box3d_multiclass_nms(
        chosen_boxes, nms_boxes, classes, cfg.score_thr, cfg.max_num, cfg, dirs[top])
    if len(kept):
        yaw = limit_period(kept[:, 6] + np.pi / 2, 0, np.pi)
        kept[:, 6] = yaw - np.pi / 2 + np.pi * kept_dirs
    return kept, final_scores, labels


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def gt_annotation(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if any(len(row) != 15 for row in rows):
        raise ValueError(f'Invalid KITTI label row: {path}')
    return dict(
        name=np.asarray([row[0] for row in rows]),
        truncated=np.asarray([float(row[1]) for row in rows]),
        occluded=np.asarray([int(row[2]) for row in rows]),
        alpha=np.asarray([float(row[3]) for row in rows]),
        bbox=np.asarray([[float(v) for v in row[4:8]] for row in rows],
                        dtype=np.float64).reshape(-1, 4),
        dimensions=np.asarray([[float(row[10]), float(row[8]), float(row[9])]
                               for row in rows], dtype=np.float64).reshape(-1, 3),
        location=np.asarray([[float(v) for v in row[11:14]] for row in rows],
                            dtype=np.float64).reshape(-1, 3),
        rotation_y=np.asarray([float(row[14]) for row in rows]),
    )


def detection_annotation(boxes, scores, labels, calib, width, height):
    """Follow pinned KittiMetric.convert_valid_bboxes and bbox2result_kitti."""
    import torch
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, points_cam2img

    lidar = LiDARInstance3DBoxes(boxes.detach().clone(), box_dim=7)
    lidar.limit_yaw(offset=.5, period=2 * np.pi)
    transform = build_kitti_lidar2cam(calib)
    camera = lidar.convert_to(Box3DMode.CAM, boxes.new_tensor(transform))
    p2 = boxes.new_tensor(calib['P2'].reshape(3, 4))
    corners = points_cam2img(camera.corners, p2)
    projected = torch.cat((corners.min(dim=1).values, corners.max(dim=1).values), dim=1)
    valid_image = ((projected[:, 0] < width) & (projected[:, 1] < height)
                   & (projected[:, 2] > 0) & (projected[:, 3] > 0))
    center = lidar.center
    lo = boxes.new_tensor([0, -39.68, -3])
    hi = boxes.new_tensor([69.12, 39.68, 1])
    valid = valid_image & ((center > lo) & (center < hi)).all(-1)
    camera_values = camera.tensor[valid].detach().cpu().numpy().astype(np.float64)
    lidar_values = lidar.tensor[valid].detach().cpu().numpy().astype(np.float64)
    xyxy = projected[valid].detach().cpu().numpy().astype(np.float64)
    xyxy[:, :2] = np.maximum(xyxy[:, :2], [0, 0])
    xyxy[:, 2:] = np.minimum(xyxy[:, 2:], [width, height])
    retained_scores = scores[valid].detach().cpu().numpy().astype(np.float64)
    retained_labels = labels[valid].detach().cpu().numpy()
    if len(retained_labels) and not np.all(retained_labels == 0):
        raise ValueError('Expected Car-only detections')
    alpha = (-np.arctan2(-lidar_values[:, 1], lidar_values[:, 0])
             + camera_values[:, 6])
    return dict(name=np.asarray(['Car'] * len(retained_scores)),
                truncated=np.zeros(len(retained_scores)),
                occluded=np.zeros(len(retained_scores), dtype=np.int64),
                alpha=alpha, bbox=xyxy,
                dimensions=camera_values[:, 3:6],
                location=camera_values[:, :3],
                rotation_y=camera_values[:, 6],
                score=retained_scores)


def evaluate(gt, dt):
    # Original eval_class supports partition counts; 4 avoids empty partitions
    # arising from the default 200 on a four-frame engineering subset.
    from mmdet3d.evaluation.functional.kitti_utils.eval import eval_class, get_mAP40

    strict = np.asarray([[[.70], [.70], [.70]]], dtype=np.float64)
    out = {}
    for kind, metric in (('2D', 0), ('BEV', 1), ('3D', 2)):
        precision = eval_class(gt, dt, [0], [0, 1, 2], metric, strict,
                               compute_aos=False, num_parts=len(gt))['precision']
        values = get_mAP40(precision)[0, :, 0]
        out[kind] = {difficulty: float(value) for difficulty, value in
                     zip(('easy', 'moderate', 'hard'), values)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pointpillars-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from mmengine.config import Config
    from mmdet3d.apis import inference_detector, init_model

    manifest = json.loads(args.manifest.read_text())
    audit = json.loads((args.scores / 'inference_audit.json').read_text())
    if (manifest['frame_ids'] != list(FRAMES) or audit['status'] != 'PASS'
            or audit['manifest_sha256'] != sha256(args.manifest)
            or [r['frame_id'] for r in audit['frames']] != list(FRAMES)
            or manifest['pointpillars_checkpoint_sha256'] != POINTPILLARS_SHA
            or audit['pointpillars_checkpoint_sha256'] != POINTPILLARS_SHA
            or audit['fusion_checkpoint_sha256'] != FUSION_SHA
            or sha256(args.pointpillars_checkpoint) != POINTPILLARS_SHA
            or sha256(args.config) != CONFIG_SHA):
        raise ValueError('Validation manifest or baseline checkpoint differs')
    cfg = Config.fromfile(str(args.config)).model.test_cfg
    if (cfg.nms_pre, cfg.nms_thr, cfg.score_thr, cfg.max_num,
            cfg.use_rotate_nms) != (100, .01, .1, 50, True):
        raise ValueError('PointPillars test settings differ')
    model = init_model(str(args.config), str(args.pointpillars_checkpoint), device='cuda:0')
    gt, predictions = [], {'pointpillars': [], 'fusion': []}
    frame_records = []
    for fid, record in zip(FRAMES, audit['frames']):
        score_path = args.scores / f'{fid}_validation_scores.npz'
        if sha256(score_path) != record['score_file_sha256']:
            raise ValueError(f'Changed score file: {fid}')
        paths = {part: args.dataset / part / f'{fid}{suffix}' for part, suffix in
                 (('label_2', '.txt'), ('calib', '.txt'), ('image_2', '.png'),
                  ('velodyne', '.bin'))}
        for part, path in paths.items():
            if sha256(path) != manifest['input_sha256'][fid][part]:
                raise ValueError(f'Changed validation input: {path}')
        gt.append(gt_annotation(paths['label_2']))
        calib = load_kitti_calibration(paths['calib'])
        with Image.open(paths['image_2']) as image:
            width, height = image.size
        with np.load(score_path, allow_pickle=False) as data:
            if str(data['frame_id'].item()) != fid:
                raise ValueError(f'Mismatched score frame: {fid}')
            boxes = data['boxes_lidar'].copy()
            directions = data['direction_labels'].copy()
            score_sets = {name: data[field].copy() for name, field in
                          (('pointpillars', 'scores_3d'), ('fusion', 'fused_scores'))}
        for name, scores in score_sets.items():
            kept, confidence, labels = postprocess(boxes, directions, scores, cfg)
            if name == 'pointpillars':
                native, _ = inference_detector(model, str(paths['velodyne']))
                reference = native.pred_instances_3d
                if (not torch.equal(kept, reference.bboxes_3d.tensor)
                        or not torch.equal(confidence, reference.scores_3d)
                        or not torch.equal(labels, reference.labels_3d)):
                    raise ValueError(f'Native detection parity failed: {fid}')
            predictions[name].append(detection_annotation(
                kept, confidence, labels, calib, width, height))
        frame_records.append(dict(frame_id=fid, gt_car=int(np.count_nonzero(
            gt[-1]['name'] == 'Car')),
            retained={key: len(predictions[key][-1]['score']) for key in predictions}))
        print(json.dumps(frame_records[-1]), flush=True)

    measured = {name: evaluate(gt, annos) for name, annos in predictions.items()}
    result = dict(schema_version='clocs_val_subset_kitti_ap40_v1', status='PASS',
                  interpretation=('Four-frame, same-frame pinned KITTI AP40 engineering '
                                  'subset; NOT full KITTI validation AP40. Too small to '
                                  'estimate dataset performance or tune parameters.'),
                  frame_ids=list(FRAMES), evaluator_partition_count=len(FRAMES),
                  source_sha256=dict(manifest=sha256(args.manifest),
                                     inference_audit=sha256(args.scores / 'inference_audit.json'),
                                     config=sha256(args.config),
                                     pointpillars_checkpoint=sha256(args.pointpillars_checkpoint)),
                  frames=frame_records, ap40_strict_iou_070=measured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', ap40_strict_iou_070=measured), indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
