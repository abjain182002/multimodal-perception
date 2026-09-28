"""Reproduce native PointPillars detections and audit four validation frames.

This is a GT coverage diagnostic, not KITTI AP: difficulty and DontCare rules
are deliberately left to the official evaluator in the next milestone.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.clocs_iou import best_camera_3d_iou

FRAMES = ('000831', '001662', '003324', '004155')
FUSION_SHA = 'a2203b9adee46b782e896b7f3b9344b5916433b5404d4bd871aa7e05d6b0c473'
POINTPILLARS_SHA = 'd42d15edce05b552fdd14e6412fc1d3e02207ee4799e6e3869f61bd30e730f3e'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def labelled_cars(path):
    """KITTI camera boxes [x,y,z,length,height,width,yaw], without ignores."""
    result = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'Car':
            h, w, length = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            result.append((x, y, z, length, h, w, float(fields[14])))
    return np.asarray(result, dtype=np.float32).reshape(-1, 7)


def postprocess(boxes, directions, scores, cfg):
    """Use pinned MMDetection3D pre-NMS top-k, rotated NMS and yaw correction."""
    import torch
    from mmdet3d.models.layers import box3d_multiclass_nms
    from mmdet3d.structures import LiDARInstance3DBoxes, limit_period, xywhr2xyxyr

    box_tensor = torch.as_tensor(boxes, device='cuda:0')
    dirs = torch.as_tensor(directions, device='cuda:0')
    values = torch.as_tensor(scores, device='cuda:0')
    if (box_tensor.shape != (107136, 7) or dirs.shape != (107136,)
            or values.shape != (107136,) or not torch.isfinite(values).all()):
        raise ValueError('Invalid saved candidate shape or score')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pointpillars-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from mmengine.config import Config
    from mmdet3d.apis import inference_detector, init_model
    from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes

    manifest = json.loads(args.manifest.read_text())
    audit_path = args.scores / 'inference_audit.json'
    audit = json.loads(audit_path.read_text())
    if (manifest['frame_ids'] != list(FRAMES) or audit['status'] != 'PASS'
            or [row['frame_id'] for row in audit['frames']] != list(FRAMES)
            or audit['manifest_sha256'] != sha256(args.manifest)
            or audit['fusion_checkpoint_sha256'] != FUSION_SHA
            or audit['pointpillars_checkpoint_sha256'] != POINTPILLARS_SHA
            or sha256(args.pointpillars_checkpoint) != POINTPILLARS_SHA):
        raise ValueError('Validation input, score, or checkpoint provenance differs')
    cfg = Config.fromfile(str(args.config)).model.test_cfg
    if (cfg.nms_pre, cfg.nms_thr, cfg.score_thr, cfg.max_num,
            cfg.use_rotate_nms) != (100, .01, .1, 50, True):
        raise ValueError('Pinned detection postprocessing settings changed')
    model = init_model(str(args.config), str(args.pointpillars_checkpoint), device='cuda:0')

    rows = []
    for fid, expected in zip(FRAMES, audit['frames']):
        score_path = args.scores / f'{fid}_validation_scores.npz'
        if sha256(score_path) != expected['score_file_sha256']:
            raise ValueError(f'Validation score file differs: {fid}')
        label_path = args.dataset / 'label_2' / f'{fid}.txt'
        calib_path = args.dataset / 'calib' / f'{fid}.txt'
        for part, path in (('label_2', label_path), ('calib', calib_path)):
            if sha256(path) != manifest['input_sha256'][fid][part]:
                raise ValueError(f'Validation {part} differs: {fid}')
        gt = labelled_cars(label_path)
        transform = build_kitti_lidar2cam(load_kitti_calibration(calib_path))
        with np.load(score_path, allow_pickle=False) as saved:
            if str(saved['frame_id'].item()) != fid:
                raise ValueError(f'Frame ID differs in scores: {fid}')
            boxes, dirs = saved['boxes_lidar'].copy(), saved['direction_labels'].copy()
            predictions = {
                'pointpillars': saved['scores_3d'].copy(),
                'fusion': saved['fused_scores'].copy(),
            }

        results = {}
        for name, scores in predictions.items():
            kept, confidences, labels = postprocess(boxes, dirs, scores, cfg)
            if name == 'pointpillars':
                native, _ = inference_detector(
                    model, str(args.dataset / 'velodyne' / f'{fid}.bin'))
                reference = native.pred_instances_3d
                if (not torch.equal(labels, reference.labels_3d)
                        or not torch.equal(confidences, reference.scores_3d)
                        or not torch.equal(kept, reference.bboxes_3d.tensor)):
                    raise ValueError(f'Native PointPillars parity failed: {fid}')
            camera = LiDARInstance3DBoxes(kept, box_dim=7).convert_to(
                Box3DMode.CAM, torch.as_tensor(transform, device='cuda:0')).tensor
            cam = camera.cpu().numpy()
            best_per_gt = best_camera_3d_iou(gt, cam)
            best_per_detection = best_camera_3d_iou(cam, gt)
            results[name] = dict(
                retained_detections=int(len(kept)),
                gt_covered_at_iou_070=int(np.count_nonzero(best_per_gt >= .70)),
                gt_best_3d_iou=[float(x) for x in best_per_gt],
                detections_below_iou_070=int(np.count_nonzero(best_per_detection < .70)),
            )
        row = dict(frame_id=fid, labelled_car=int(len(gt)), methods=results)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = dict(schema_version='clocs_val_smoke_gt_coverage_v1', status='PASS',
                  interpretation=('Four-frame best-IoU diagnostic against all labelled Car; '
                                  'no difficulty filtering, DontCare suppression, one-to-one '
                                  'matching, or official KITTI AP40.'),
                  input_sha256=dict(manifest=sha256(args.manifest),
                                    inference_audit=sha256(audit_path),
                                    config=sha256(args.config)),
                  native_pointpillars_parity_frames=4, frames=rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gt_coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - .27, [r['labelled_car'] for r in rows], .27, label='Labelled Car')
    for offset, key, label in ((0, 'pointpillars', 'PointPillars'),
                               (.27, 'fusion', 'CLOCs rescoring')):
        ax.bar(x + offset, [r['methods'][key]['gt_covered_at_iou_070'] for r in rows],
               .27, label=label)
    ax.set_xticks(x, FRAMES)
    ax.set_ylabel('Cars with any retained 3D box at IoU >= 0.70')
    ax.set_title('Four-frame best-IoU coverage diagnostic (not KITTI AP40)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'gt_coverage.png', dpi=180)
    plt.close(fig)
    print('Saved:', args.output / 'gt_coverage.json', args.output / 'gt_coverage.png')


if __name__ == '__main__':
    main()
