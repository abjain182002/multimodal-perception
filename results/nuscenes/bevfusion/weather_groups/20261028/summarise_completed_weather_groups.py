"""Verify the completed CPU evidence ZIP and build a coverage-aware figure.

No inference or ground-truth replay. AP arithmetic is recomputed from saved PR
curves, and full replay JSONs are reconciled with the preserved references.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import zipfile

from evaluate_weather_groups import compare_metrics, GROUPS, MODELS

EXPECTED_SHA = '75126afda2354f95f9402fccfa37100fb45d3f956cb1d5600cfb6fdf2c289227'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'completed')
    args = p.parse_args()
    assert args.archive.stat().st_size == 2156223
    assert sha(args.archive.read_bytes()) == EXPECTED_SHA
    args.output.mkdir(parents=True, exist_ok=True)
    numeric_checks, curve_checks, maximum_error = 0, 0, 0.
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('evidence_manifest.json'))['files']
        assert len(manifest) == 69
        assert set(z.namelist()) == set(manifest) | {'evidence_manifest.json'}
        for name, record in manifest.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and sha(data) == record['sha256'], name
        load = lambda name: json.loads(z.read(name))
        rows = list(csv.DictReader(io.StringIO(z.read('group_metrics.csv').decode())))
        classes = list(csv.DictReader(io.StringIO(z.read('class_metrics.csv').decode())))
        assert len(rows) == 8 and len(classes) == 80
        summaries, gt_counts = {}, {}
        for model in MODELS:
            audit = load(f'reference/{model}/evaluation_audit.json')
            for name, key in [('metrics_summary.json', 'metrics_summary_sha256'),
                              ('metrics_details.json', 'metrics_details_sha256')]:
                assert sha(z.read(f'reference/{model}/{name}')) == audit[key]
                errors, maximum, count = compare_metrics(load(f'{model}/full_validation/{name}'),
                                                         load(f'reference/{model}/{name}'))
                assert not errors, errors[:3]
                maximum_error = max(maximum_error, maximum); numeric_checks += count
            assert load(f'{model}/full_replay_check.json')['status'] == 'PASS'
            partition_gt = Counter()
            for group in ('full_validation', *GROUPS):
                s = load(f'{model}/{group}/metrics_summary.json')
                details = load(f'{model}/{group}/metrics_details.json')
                coverage = load(f'{model}/{group}/coverage.json')
                cfg = s['cfg']
                for c, aps in s['label_aps'].items():
                    for distance, expected in aps.items():
                        precision = details[c + ':' + distance]['precision']
                        assert len(precision) == 101
                        trimmed = precision[round(100 * cfg['min_recall']) + 1:]
                        actual = sum(max(0., x - cfg['min_precision']) for x in trimmed) / len(trimmed) / (1 - cfg['min_precision'])
                        assert abs(actual - expected) < 1e-12
                        curve_checks += 1
                    assert abs(sum(aps.values()) / len(aps) - s['mean_dist_aps'][c]) < 1e-12
                assert abs(sum(s['mean_dist_aps'].values()) / 10 - s['mean_ap']) < 1e-12
                nds = (cfg['mean_ap_weight'] * s['mean_ap'] + sum(s['tp_scores'].values())) / (cfg['mean_ap_weight'] + len(s['tp_scores']))
                assert abs(nds - s['nd_score']) < 1e-12
                if group == 'full_validation':
                    continue
                summaries[model, group] = s
                gt_counts[model, group] = coverage['filtered_gt_counts']
                partition_gt.update(coverage['filtered_gt_counts'])
                row = next(r for r in rows if r['model'] == model and r['group'] == group)
                assert abs(float(row['mAP_x100']) - 100 * s['mean_ap']) < 1e-12
                assert abs(float(row['NDS_x100']) - 100 * s['nd_score']) < 1e-12
                for c, count in coverage['filtered_gt_counts'].items():
                    cr = next(r for r in classes if r['model'] == model and r['group'] == group and r['class'] == c)
                    assert int(cr['filtered_GT']) == count
                    assert abs(float(cr['AP_x100']) - 100 * s['mean_dist_aps'][c]) < 1e-12
                    if not count: assert s['mean_dist_aps'][c] == 0
            assert dict(partition_gt) == audit['filtered_gt_counts']
        for group in GROUPS:
            assert gt_counts['lidar', group] == gt_counts['lidar_camera', group]
        common = [c for c in gt_counts['lidar', GROUPS[0]] if all(gt_counts['lidar', g][c] > 0 for g in GROUPS)]
        assert set(common) == {'car', 'truck', 'pedestrian', 'motorcycle'}
        regressions = []
        for row in rows:
            model, group = row['model'], row['group']; s = summaries[model, group]
            represented = [c for c in gt_counts[model, group] if gt_counts[model, group][c] > 0]
            for column, chosen in [('represented_classes_macro_AP_x100', represented), ('common_classes_macro_AP_x100', common)]:
                assert abs(float(row[column]) - 100 * sum(s['mean_dist_aps'][c] for c in chosen) / len(chosen)) < 1e-12
        paired = load('paired_summary.json')
        for group in GROUPS:
            for key in ('mean_ap', 'nd_score'):
                difference = 100 * (summaries['lidar_camera', group][key] - summaries['lidar', group][key])
                assert abs(difference - paired['paired_fusion_minus_lidar_points'][group][key]) < 1e-12
            for c, count in gt_counts['lidar', group].items():
                difference = 100 * (summaries['lidar_camera', group]['mean_dist_aps'][c] - summaries['lidar', group]['mean_dist_aps'][c])
                if count > 0 and difference < 0:
                    regressions.append({'group': group, 'class': c, 'GT_instances': count, 'fusion_minus_lidar_AP_points': difference})
        selected = ['group_metrics.csv', 'class_metrics.csv', 'paired_summary.json', 'evidence_manifest.json',
                    'evaluation_metadata_derivation.json', 'description_review.json',
                    'lidar/full_replay_check.json', 'lidar_camera/full_replay_check.json']
        for name in selected:
            target = args.output / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
        archive_target = args.output / args.archive.name
        shutil.copyfile(args.archive, archive_target)
        assert sha(archive_target.read_bytes()) == EXPECTED_SHA
    verification = {'status': 'WEATHER_GROUP_COMPLETED_EVIDENCE_IMPORT_AND_ARITHMETIC_PASS',
        'archive': args.archive.name, 'bytes': args.archive.stat().st_size, 'sha256': EXPECTED_SHA,
        'verified_payload_members': 69, 'full_replay_numeric_values_reconciled': numeric_checks,
        'maximum_full_replay_absolute_error': maximum_error,
        'AP_values_recomputed_from_saved_PR_curves': curve_checks,
        'group_rows_checked': 8, 'per_class_rows_checked': 80,
        'group_GT_counts_partition_full_validation': True, 'same_GT_counts_both_models': True,
        'common_classes': common, 'represented_class_regressions': regressions,
        'Drive_provenance': 'Colab final receipt reports verified Drive copy; uploaded ZIP matches that reported SHA-256. Drive was not directly accessed during local import.',
        'scope': 'Archive verification, recorded replay/reference JSON reconciliation, curve-to-AP arithmetic and CSV consistency. No fresh inference or local GT evaluation.'}
    write_json(args.output / 'import_verification.json', verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharey=True)
    x = np.arange(4); labels = ['Neither\nmentioned', 'Rain only\nmentioned', 'Night only\nmentioned', 'Night + rain\nmentioned']
    titles = ['SDK mAP: 10 classes', 'Macro AP: same 4 classes', 'Pedestrian AP']
    for index, ax in enumerate(axes):
        for offset, model, label, color in [(-.19, 'lidar', 'LiDAR pipeline', '#356899'), (.19, 'lidar_camera', 'Fusion pipeline', '#d67a25')]:
            values = []
            for g in GROUPS:
                s = summaries[model, g]
                value = (s['mean_ap'] if index == 0 else sum(s['mean_dist_aps'][c] for c in common) / 4
                         if index == 1 else s['mean_dist_aps']['pedestrian'])
                values.append(100 * value)
            bars = ax.bar(x + offset, values, .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0, 103)
        ax.set_title(titles[index], fontsize=11); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Score x100'); axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Natural scene-description groups: scores and class coverage', fontsize=13)
    fig.text(.025, .105, 'Groups: 111 / 24 / 12 / 3 scenes; 4,449 / 968 / 482 / 120 frames. All 15 night scenes share one acquisition log.', fontsize=8)
    fig.text(.025, .068, 'Same 4 classes: car, truck, pedestrian, motorcycle. Pedestrian GT instances across frames: 22,411 / 1,003 / 64 / 22.', fontsize=8)
    fig.text(.025, .031, 'Night-only / night + rain have 3 / 6 absent GT classes. Descriptive estimates; no weather-causality or statistical-significance claim.', fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(args.output / ('weather_coverage_comparison.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
