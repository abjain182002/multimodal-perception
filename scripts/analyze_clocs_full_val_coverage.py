"""Describe paired Car 3D-IoU coverage across the frozen validation frames.

This is an all-labelled-Car diagnostic. It does not implement KITTI difficulty,
DontCare, one-to-one matching, or AP; see the paired AP40 audit separately.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from fusion.clocs_iou import best_camera_3d_iou
from scripts.record_clocs_full_val_paired import verify_archive

THRESHOLD = .70
RANGES = ('<20 m', '20–40 m', '40–60 m', '60+ m')


def camera_boxes(anno, indices):
    location = np.asarray(anno['location'], dtype=np.float32).reshape(-1, 3)
    dimensions = np.asarray(anno['dimensions'], dtype=np.float32).reshape(-1, 3)
    yaw = np.asarray(anno['rotation_y'], dtype=np.float32).reshape(-1, 1)
    return np.concatenate((location, dimensions, yaw), axis=1)[indices]


def range_name(depth):
    if depth < 20:
        return RANGES[0]
    if depth < 40:
        return RANGES[1]
    if depth < 60:
        return RANGES[2]
    return RANGES[3]


def fraction(numerator, denominator):
    return round(100 * numerator / denominator, 4) if denominator else None


def make_figure(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(RANGES))
    baseline = [report['by_range'][key]['pointpillars_coverage_percent']
                for key in RANGES]
    fusion = [report['by_range'][key]['fusion_coverage_percent']
              for key in RANGES]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.bar(x - .18, baseline, .36, label='PointPillars (this run)')
    ax.bar(x + .18, fusion, .36, label='CLOCs (this run)')
    ax.set_xticks(x, RANGES)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Annotated Car depth')
    ax.set_ylabel('Cars with any retained box at 3D IoU ≥ 0.70 (%)')
    ax.set_title('Full KITTI validation: best-IoU coverage by range')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.text(.5, .014,
             'All labelled Cars; includes ignored cases and duplicates; not KITTI AP40.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(directory / 'coverage_by_range.png', dpi=180)
    fig.savefig(directory / 'coverage_by_range.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/clocs_full_val_paired'))
    args = parser.parse_args()
    _, paired, ordered_digest = verify_archive(args.archive)
    totals = defaultdict(lambda: {'cars': 0, 'pointpillars': 0, 'fusion': 0,
                                  'rescued': 0, 'lost': 0})
    frame_rows = []
    with ZipFile(args.archive) as archive:
        ids = json.loads(archive.read('inputs/manifest.json'))['frame_ids']
        for fid in ids:
            frame = json.loads(archive.read(f'predictions/{fid}.json'))
            gt = frame['ground_truth']
            car_indices = [i for i, name in enumerate(gt['name']) if name == 'Car']
            cars = camera_boxes(gt, car_indices)
            predictions = {method: camera_boxes(frame['detections'][method],
                            slice(None)) for method in ('pointpillars', 'fusion')}
            best = {method: best_camera_3d_iou(cars, boxes)
                    for method, boxes in predictions.items()}
            matched = {method: best[method] >= THRESHOLD for method in best}
            rescued = matched['fusion'] & ~matched['pointpillars']
            lost = matched['pointpillars'] & ~matched['fusion']
            for i, car in enumerate(cars):
                strata = ('all', range_name(float(car[2])),
                          f'occlusion_{gt["occluded"][car_indices[i]]}')
                for name in strata:
                    entry = totals[name]
                    entry['cars'] += 1
                    entry['pointpillars'] += int(matched['pointpillars'][i])
                    entry['fusion'] += int(matched['fusion'][i])
                    entry['rescued'] += int(rescued[i])
                    entry['lost'] += int(lost[i])
            frame_rows.append(dict(
                frame_id=fid, labelled_car=len(cars),
                pointpillars_detections=len(predictions['pointpillars']),
                fusion_detections=len(predictions['fusion']),
                pointpillars_covered=int(matched['pointpillars'].sum()),
                fusion_covered=int(matched['fusion'].sum()),
                rescued_cars=int(rescued.sum()), lost_cars=int(lost.sum())))

    def summarized(key):
        item = dict(totals[key])
        item['pointpillars_coverage_percent'] = fraction(item['pointpillars'], item['cars'])
        item['fusion_coverage_percent'] = fraction(item['fusion'], item['cars'])
        return item

    report = {
        'schema_version': 'clocs_full_val_best_iou_coverage_v1',
        'status': 'PASS',
        'interpretation': ('Best 3D IoU of each labelled Car against any retained '
                           'detection; no KITTI difficulty or DontCare handling, '
                           'no one-to-one matching, and not AP40. Historical baseline '
                           'parity remains unresolved.'),
        'frames': len(frame_rows), 'iou_threshold': THRESHOLD,
        'ordered_prediction_digests': ordered_digest,
        'paired_audit_status': paired['status'],
        'all_labelled_car': summarized('all'),
        'by_range': {name: summarized(name) for name in RANGES},
        'by_occlusion': {f'occlusion_{i}': summarized(f'occlusion_{i}')
                         for i in range(4)},
        'frames_with_rescue': sum(r['rescued_cars'] > 0 for r in frame_rows),
        'frames_with_loss': sum(r['lost_cars'] > 0 for r in frame_rows),
        'largest_rescues': sorted(frame_rows, key=lambda r: (-r['rescued_cars'],
                                                             r['frame_id']))[:12],
        'largest_losses': sorted(frame_rows, key=lambda r: (-r['lost_cars'],
                                                            r['frame_id']))[:12],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'coverage_diagnostic.json').write_text(
        json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'coverage_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(frame_rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(frame_rows)
    make_figure(report, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        'status', 'frames', 'all_labelled_car', 'by_range',
        'frames_with_rescue', 'frames_with_loss',
        'largest_rescues', 'largest_losses')}, indent=2))


if __name__ == '__main__':
    main()
