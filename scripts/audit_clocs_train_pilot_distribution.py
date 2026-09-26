#!/usr/bin/env python3
"""Inspect and plot frozen train-only CLOCs pilot labels and associations."""

import argparse
import json
from pathlib import Path

import numpy as np

from fusion.clocs_dataset import load_training_example_npz
from scripts.export_clocs_dense_train_frame import file_sha256


SCORE_EDGES = np.linspace(0.0, 1.0, 21)
RANGE_EDGES = np.array([20.0, 40.0, 60.0])
RANGE_NAMES = ('0–20', '20–40', '40–60', '60+')
LABEL_NAMES = ('negative', 'ignored', 'positive')


def inspect_frame(frame, root):
    fid = frame['frame_id']
    example_file = root / f'{fid}_real_targets.npz'
    rows_file = root / f'{fid}_clocs_rows.npz'
    candidate_file = root / f'{fid}_dense_candidates.npz'
    if file_sha256(candidate_file) != frame['candidate_sha256']:
        raise ValueError(f'Candidate hash mismatch: {fid}')
    if file_sha256(rows_file) != frame['rows_sha256']:
        raise ValueError(f'Feature-row hash mismatch: {fid}')
    example = load_training_example_npz(example_file, verify_integrity=True)
    if (example['frame_id'] != fid or example['split'] != 'train'
            or example['content_sha256'] != frame['training_example_content_sha256']):
        raise ValueError(f'Training example changed: {fid}')
    with np.load(candidate_file, allow_pickle=False) as candidate, np.load(rows_file, allow_pickle=False) as rows:
        if (str(candidate['frame_id'].item()) != fid
                or str(rows['frame_id'].item()) != fid
                or not np.array_equal(candidate['scores_3d'], example['candidate_scores_3d'])
                or not np.array_equal(rows['features'], example['feature_rows'])
                or not np.array_equal(rows['candidate_index'], example['candidate_indices'])):
            raise ValueError(f'Saved feature/candidate alignment changed: {fid}')
        normalized_distance = np.array(candidate['distance_82'], copy=True)
        row_types = np.array(rows['row_type'], copy=True)
        row_index = np.array(rows['candidate_index'], copy=True)

    labels = example['labels']
    scores = example['candidate_scores_3d']
    n = len(labels)
    if (n != frame['num_candidates'] or len(normalized_distance) != n
            or len(row_types) != example['num_rows']
            or not np.isfinite(normalized_distance).all()
            or np.any(normalized_distance < 0)
            or not np.isin(row_types, [0, 1]).all()):
        raise ValueError(f'Unexpected candidate/row values: {fid}')
    per_class = {key: int(np.count_nonzero(labels == value))
                 for key, value in (('positive', 1), ('negative', 0), ('ignored', -1))}
    for label in LABEL_NAMES:
        if per_class[label] != frame['num_' + label]:
            raise ValueError(f'Candidate-label count changed: {fid}/{label}')

    matched = np.zeros(n, dtype=bool)
    matched[row_index[row_types == 1]] = True
    if int(np.count_nonzero(row_types == 0)) != int(np.count_nonzero(~matched)):
        raise ValueError(f'Sentinel coverage differs from unmatched candidates: {fid}')
    range_bin = np.digitize(normalized_distance * 82.0, RANGE_EDGES)
    by_range = {label: np.bincount(range_bin[labels == value], minlength=len(RANGE_NAMES)).astype(int)
                for label, value in (('negative', 0), ('ignored', -1), ('positive', 1))}
    by_score = {label: np.histogram(scores[labels == value], bins=SCORE_EDGES)[0].astype(int)
                for label, value in (('negative', 0), ('ignored', -1), ('positive', 1))}
    return {
        'frame_id': fid, 'gt': frame['car_gt_after_range_filter'],
        'yolo': frame['yolov8x_car_detections'],
        'counts': per_class, 'feature_rows': example['num_rows'],
        'matched_candidate_count': int(matched.sum()),
        'matched_positive_count': int(np.count_nonzero(matched & (labels == 1))),
        'matched_row_count': int(np.count_nonzero(row_types == 1)),
        'unmatched_row_count': int(np.count_nonzero(row_types == 0)),
        'score_hist': {k: v.tolist() for k, v in by_score.items()},
        'range_hist': {k: v.tolist() for k, v in by_range.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', default='results/clocs_train_16_training_example_audit.json')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='results/clocs_train_16_distribution_audit.json')
    parser.add_argument('--png', default='results/clocs_train_16_distribution.png')
    parser.add_argument('--pdf', default='results/clocs_train_16_distribution.pdf')
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    if (summary['schema_version'] != 'clocs_train_16_training_example_audit_v1'
            or summary['status'] != 'PASS' or len(summary['frames']) != 16
            or summary['completed_frames'] != 16):
        raise ValueError('Expected the verified 16-frame training pilot')
    records = [inspect_frame(frame, Path(args.data_dir)) for frame in summary['frames']]
    counts = {label: sum(f['counts'][label] for f in records) for label in LABEL_NAMES}
    if (counts['positive'] != summary['totals']['num_positive']
            or counts['negative'] != summary['totals']['num_negative']
            or counts['ignored'] != summary['totals']['num_ignored']
            or sum(f['feature_rows'] for f in records) != summary['totals']['num_feature_rows']):
        raise ValueError('Distribution does not reproduce the saved batch totals')

    score_hist = {label: np.sum([f['score_hist'][label] for f in records], axis=0)
                  for label in LABEL_NAMES}
    range_hist = {label: np.sum([f['range_hist'][label] for f in records], axis=0)
                  for label in LABEL_NAMES}
    if any(int(score_hist[k].sum()) != counts[k] or int(range_hist[k].sum()) != counts[k]
           for k in LABEL_NAMES):
        raise ValueError('Histogram totals differ from candidate labels')
    num_candidates = sum(counts.values())
    num_matched = sum(f['matched_candidate_count'] for f in records)
    num_matched_pos = sum(f['matched_positive_count'] for f in records)
    result = {
        'schema_version': 'clocs_train_16_distribution_audit_v1',
        'status': 'PASS', 'split': 'train',
        'interpretation': 'Deliberately GT-stratified 16-frame pilot; descriptive labels, not detector AP or population prevalence.',
        'source_batch_audit_sha256': file_sha256(args.summary),
        'num_frames': 16, 'num_candidates': num_candidates,
        'num_feature_rows': sum(f['feature_rows'] for f in records),
        'num_car_gt': sum(f['gt'] for f in records),
        'num_yolov8x_car_detections': sum(f['yolo'] for f in records),
        'label_counts': counts,
        'positive_fraction_of_candidates': counts['positive'] / num_candidates,
        'negative_to_positive_ratio': counts['negative'] / counts['positive'],
        'frames_without_retained_car_gt': sum(f['gt'] == 0 for f in records),
        'frames_with_gt_without_positive': [f['frame_id'] for f in records
                                            if f['gt'] > 0 and not f['counts']['positive']],
        'camera_association': {
            'matched_candidates': num_matched,
            'unmatched_candidates': num_candidates - num_matched,
            'matched_positive_candidates': num_matched_pos,
            'positive_candidates_without_camera_match': counts['positive'] - num_matched_pos,
            'matched_rows': sum(f['matched_row_count'] for f in records),
            'unmatched_sentinel_rows': sum(f['unmatched_row_count'] for f in records),
        },
        'score_histogram': {'edges': SCORE_EDGES.tolist(),
                            **{k: v.tolist() for k, v in score_hist.items()}},
        'lidar_xy_range_histogram_m': {'bins': list(RANGE_NAMES),
                                       **{k: v.tolist() for k, v in range_hist.items()}},
        'frames': [{k: f[k] for k in ('frame_id', 'gt', 'yolo', 'counts', 'feature_rows',
                                     'matched_candidate_count', 'matched_positive_count')}
                   for f in records],
    }
    if counts['positive'] == 0:
        raise ValueError('No positive signal in pilot')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    fig.suptitle('CLOCs training data audit | 16 selected KITTI frames', fontsize=16, weight='bold')
    fig.text(0.5, 0.96, 'Train-only, GT-stratified engineering pilot; these distributions do not estimate validation accuracy',
             ha='center', fontsize=9, color='#4b5563')

    ax = axes[0, 0]
    ax.scatter([f['gt'] for f in records], [f['counts']['positive'] for f in records],
               s=65, color='#2563eb', alpha=0.85)
    for f in records:
        if f['gt'] >= 7 or (f['gt'] and f['counts']['positive'] < 15):
            offset = {'000149': (-52, -18), '004842': (6, 9)}.get(f['frame_id'], (5, 4))
            ax.annotate(f['frame_id'], (f['gt'], f['counts']['positive']),
                        xytext=offset, textcoords='offset points', fontsize=8)
    ax.set(xlabel='Retained Car GT per frame', ylabel='Positive 3D candidates',
           title='Positive training signal by scene')
    ax.grid(alpha=0.18)

    ax = axes[0, 1]
    centers = (SCORE_EDGES[:-1] + SCORE_EDGES[1:]) / 2
    for label, color in (('positive', '#ea580c'), ('negative', '#2563eb'), ('ignored', '#7c3aed')):
        ax.step(centers, score_hist[label] / counts[label], where='mid',
                label=f'{label} (n={counts[label]:,})', color=color, linewidth=2)
    ax.set(xlabel='PointPillars Car score', ylabel='Fraction within each label',
           title='Score distribution by target label', xlim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[1, 0]
    total_by_range = sum(range_hist.values())
    positive_per_1000 = np.divide(range_hist['positive'] * 1000, total_by_range,
                                  out=np.zeros(len(RANGE_NAMES), dtype=float), where=total_by_range > 0)
    bars = ax.bar(RANGE_NAMES, positive_per_1000, color='#0d9488')
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in positive_per_1000], padding=3, fontsize=8)
    ax.set(xlabel='Candidate LiDAR xy range (m)',
           ylabel='Positive labels per 1,000 candidates',
           title='Label prevalence by candidate range')
    ax.grid(axis='y', alpha=0.18)

    ax = axes[1, 1]
    order = ('positive', 'ignored', 'negative')
    bars = ax.bar(order, [counts[k] - 1 for k in order], bottom=1,
                  color=['#ea580c', '#7c3aed', '#2563eb'])
    ax.set_yscale('log')
    ax.set_ylim(1, 5_000_000)
    ax.set(ylabel='Number of candidates (log scale)', title='Target imbalance')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.bar_label(bars, labels=[f'{counts[k]:,}' for k in order], padding=3, fontsize=9)
    ax.grid(axis='y', alpha=0.18)

    for path in (args.output, args.png, args.pdf):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    fig.savefig(args.png, dpi=220)
    fig.savefig(args.pdf)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ('status', 'num_frames', 'num_candidates',
                                           'num_feature_rows', 'label_counts',
                                           'positive_fraction_of_candidates',
                                           'negative_to_positive_ratio',
                                           'camera_association')}, indent=2))
    print('Saved audit and figures:', args.output, args.png, args.pdf)


if __name__ == '__main__':
    main()
