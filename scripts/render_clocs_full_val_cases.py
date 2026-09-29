"""Render paired validation rescue/loss scenes from verified predictions.

These image overlays show 2D projections of 3D detections, while the marked
Car changes are defined using rotated 3D IoU >= 0.70. They are diagnostics,
not a KITTI AP evaluation or one-to-one assignment.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from fusion.clocs_iou import best_camera_3d_iou
from scripts.analyze_clocs_full_val_coverage import camera_boxes
from scripts.record_clocs_full_val_paired import verify_archive

CASES = ('001583', '000393', '000831')


def save_case(fid, frame, image_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    ground_truth = frame['ground_truth']
    car_indices = [i for i, name in enumerate(ground_truth['name']) if name == 'Car']
    gt_boxes = camera_boxes(ground_truth, car_indices)
    gt_bboxes = np.asarray(ground_truth['bbox'], dtype=float).reshape(-1, 4)
    predictions = frame['detections']
    best = {}
    for name in ('pointpillars', 'fusion'):
        best[name] = best_camera_3d_iou(
            gt_boxes, camera_boxes(predictions[name], slice(None)))
    rescued = (best['fusion'] >= .7) & (best['pointpillars'] < .7)
    lost = (best['pointpillars'] >= .7) & (best['fusion'] < .7)

    with Image.open(image_path) as source:
        image = np.asarray(source.convert('RGB'))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.3), sharex=True, sharey=True)
    records = []
    for axis, name in zip(axes, ('pointpillars', 'fusion')):
        axis.imshow(image)
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.axis('off')
        label = 'PointPillars' if name == 'pointpillars' else 'CLOCs rescoring'
        axis.set_title(f'{label} | {len(predictions[name]["score"])} retained boxes',
                       fontsize=12)
        for idx in car_indices:
            x0, y0, x1, y1 = gt_bboxes[idx]
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     fill=False, edgecolor='#16c1d5',
                                     linewidth=1.0, alpha=.7))
        boxes_2d = np.asarray(predictions[name]['bbox'], dtype=float).reshape(-1, 4)
        for box in boxes_2d:
            x0, y0, x1, y1 = box
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     fill=False, edgecolor='#f3a24b',
                                     linewidth=1.2, alpha=.75))
        for local, gt_index in enumerate(car_indices):
            if not (rescued[local] or lost[local]):
                continue
            x0, y0, x1, y1 = gt_bboxes[gt_index]
            color = '#38c962' if rescued[local] else '#ec5363'
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     fill=False, edgecolor=color, linewidth=2.7))
    for local, gt_index in enumerate(car_indices):
        if rescued[local] or lost[local]:
            records.append(dict(
                gt_index=gt_index, change='rescued' if rescued[local] else 'lost',
                depth_m=float(gt_boxes[local, 2]),
                pointpillars_best_3d_iou=float(best['pointpillars'][local]),
                fusion_best_3d_iou=float(best['fusion'][local])))
    fig.suptitle(f'KITTI {fid} | cyan GT, orange detections, green rescue, red loss',
                 fontsize=13)
    fig.text(.5, .015,
             '2D box overlays; rescue/loss defined by 3D IoU ≥ 0.70. Not KITTI AP40.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(output / f'case_{fid}.png', dpi=170)
    fig.savefig(output / f'case_{fid}.pdf')
    plt.close(fig)
    return dict(frame_id=fid, labelled_car=len(car_indices),
                pointpillars_detections=len(predictions['pointpillars']['score']),
                fusion_detections=len(predictions['fusion']['score']),
                rescued=int(rescued.sum()), lost=int(lost.sum()),
                changed_cars=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/clocs_full_val_paired/cases'))
    args = parser.parse_args()
    _, paired, ordered = verify_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    with ZipFile(args.archive) as archive:
        for fid in CASES:
            path = args.image_dir / f'{fid}.png'
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = json.loads(archive.read(f'predictions/{fid}.json'))
            cases.append(save_case(fid, frame, path, args.output_dir))
    if [(c['rescued'], c['lost']) for c in cases] != [(3, 0), (0, 3), (2, 1)]:
        raise ValueError('Case selection differs from the verified coverage audit')
    report = dict(schema_version='clocs_full_val_case_studies_v1', status='PASS',
                  interpretation=('Visualized diagnostic rescue/loss boxes on three '
                                  'validation frames; not KITTI AP or proof of cause.'),
                  ordered_prediction_digests=ordered,
                  paired_audit_status=paired['status'], cases=cases)
    (args.output_dir / 'case_studies.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
