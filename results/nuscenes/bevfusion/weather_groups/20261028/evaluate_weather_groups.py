"""Pinned nuScenes CPU evaluator and descriptive weather-group report.

Called by run_weather_groups_colab.py, once per model in fresh processes.
No inference. Preserves official token iteration order when selecting groups.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import sys


MODELS = ('lidar', 'lidar_camera')
GROUPS = ('neither_mentioned', 'rain_only_mentioned',
          'night_only_mentioned', 'night_and_rain_mentioned')
PREDICTION_HASHES = {
    'lidar': '15667405cb1af580e6f8082dcc50366560e45335c36f7122ec490f377cd86c63',
    'lidar_camera': 'dcd86893985366881b27d3841b64b93a85988158b518f0c9edfa274c291596dd',
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    # SDK summaries/details contain legitimate undefined class-specific NaNs.
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def compare_metrics(actual, expected, tolerance=1e-12, path='root'):
    """Recursive same-evaluator replay check; only eval_time is excluded."""
    errors, maximum, count = [], 0., 0
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [path + ': keys/type mismatch'], maximum, count
        children = [(actual[k], v, path + '.' + str(k)) for k, v in expected.items() if k != 'eval_time']
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path + ': list mismatch'], maximum, count
        children = [(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(actual, expected))]
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return [path + ': numeric type mismatch'], maximum, count
        count = 1
        if math.isnan(expected):
            if not math.isnan(actual):
                errors.append(path + ': NaN mismatch')
        elif not math.isfinite(expected):
            if actual != expected:
                errors.append(path + ': infinity mismatch')
        elif not math.isfinite(actual):
            errors.append(path + ': nonfinite actual')
        else:
            maximum = abs(actual - expected)
            if maximum > tolerance:
                errors.append(path + ': numeric difference ' + str(maximum))
        return errors, maximum, count
    else:
        return ([] if actual == expected else [path + ': value mismatch']), maximum, count
    for a, b, subpath in children:
        e, m, n = compare_metrics(a, b, tolerance, subpath)
        errors.extend(e)
        maximum = max(maximum, m)
        count += n
    return errors, maximum, count


def subset_boxes(boxes, tokens):
    from nuscenes.eval.common.data_classes import EvalBoxes
    assert tokens <= set(boxes.sample_tokens)
    subset = EvalBoxes()
    # Retain full evaluator order, including samples with zero boxes.
    for t in boxes.sample_tokens:
        if t in tokens:
            subset.add_boxes(t, boxes[t])
    assert set(subset.sample_tokens) == tokens
    return subset


def save_evaluation(evaluator, directory):
    directory.mkdir(parents=True, exist_ok=False)
    metrics, details = evaluator.evaluate()
    summary = metrics.serialize()
    summary['meta'] = evaluator.meta.copy()
    write_json(directory / 'metrics_summary.json', summary)
    write_json(directory / 'metrics_details.json', details.serialize())
    classes = list(evaluator.cfg.class_names)
    gt = Counter(b.detection_name for b in evaluator.gt_boxes.all)
    pred = Counter(b.detection_name for b in evaluator.pred_boxes.all)
    ledger = {'samples': len(evaluator.sample_tokens),
              'filtered_gt_counts': {c: gt[c] for c in classes},
              'filtered_prediction_counts': {c: pred[c] for c in classes},
              'absent_gt_classes': [c for c in classes if not gt[c]],
              'gt_classes_below_20_instances': [c for c in classes if 0 < gt[c] < 20]}
    write_json(directory / 'coverage.json', ledger)
    return summary, ledger


def evaluate_model(args):
    import nuscenes
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.splits import create_splits_scenes

    work = args.work
    environment = json.loads((work / 'reference/environment_audit.json').read_text())
    for name, version in environment['packages'].items():
        assert metadata.version(name) == version, name
    sdk = Path(nuscenes.__file__).parent
    for relative, expected in environment['evaluator_source_sha256'].items():
        assert file_sha(sdk / relative) == expected, relative
    write_json(work / f'{args.model}_environment.json', {
        'python': platform.python_version(), 'packages': {n: metadata.version(n) for n in environment['packages']},
        'evaluator_source_sha256': environment['evaluator_source_sha256'],
        'source_hashes_verified': True, 'worker_sha256': file_sha(__file__)})
    with (work / 'preparation/sample_membership.csv').open() as f:
        membership = list(csv.DictReader(f))
    tokens = {r['sample_token'] for r in membership}
    assert len(tokens) == len(membership) == 6019
    pred = args.base / 'full_validation_merged' / args.model / 'results_nusc.json'
    print('VERIFYING FULL SAVED PREDICTIONS:', args.model, flush=True)
    assert file_sha(pred) == PREDICTION_HASHES[args.model]
    cfg = config_factory('detection_cvpr_2019')
    assert cfg.serialize() == json.loads((work / 'reference/detection_config.json').read_text())
    print('LOADING EVALUATION-ONLY METADATA:', args.model, flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.root), verbose=True)
    official = set(create_splits_scenes()['val'])
    official_tokens = {s['token'] for s in nusc.sample if nusc.get('scene', s['scene_token'])['name'] in official}
    assert official_tokens == tokens
    scene_names = {r['scene_name'] for r in membership}
    assert scene_names == official and len(scene_names) == 150
    out = work / args.model
    out.mkdir(exist_ok=False)
    evaluator = NuScenesEval(nusc=nusc, config=cfg, result_path=str(pred), eval_set='val',
                            output_dir=str(out / 'sdk_initialization'), verbose=True)
    assert set(evaluator.pred_boxes.sample_tokens) == set(evaluator.gt_boxes.sample_tokens) == tokens
    full_gt, full_pred = evaluator.gt_boxes, evaluator.pred_boxes
    full_tokens = evaluator.sample_tokens
    print('FULL-SPLIT REPLAY BEFORE SUBGROUPS:', args.model, flush=True)
    summary, ledger = save_evaluation(evaluator, out / 'full_validation')
    original_audit = json.loads((work / f'reference/{args.model}/evaluation_audit.json').read_text())
    assert ledger['filtered_gt_counts'] == original_audit['filtered_gt_counts']
    differences, max_error, checked = [], 0., 0
    for name in ('metrics_summary.json', 'metrics_details.json'):
        actual = json.loads((out / 'full_validation' / name).read_text())
        expected = json.loads((work / 'reference' / args.model / name).read_text())
        e, m, n = compare_metrics(actual, expected)
        differences.extend(e); max_error = max(max_error, m); checked += n
    write_json(out / 'full_replay_check.json', {
        'status': 'PASS' if not differences else 'FAIL', 'tolerance': 1e-12,
        'maximum_absolute_error': max_error, 'numeric_values_checked': checked,
        'differences': differences[:30], 'difference_count': len(differences),
        'excluded': ['eval_time'], 'prediction_sha256': PREDICTION_HASHES[args.model],
        'scope': 'Same-evaluator full-split replay, also validating evaluation-only metadata against preserved original metrics.'})
    assert not differences, 'Full replay differs: subgroup evaluation stopped; evidence retained.'
    for group in GROUPS:
        chosen = {r['sample_token'] for r in membership if r['group'] == group}
        assert chosen, 'Empty group; revise report handling before evaluation'
        evaluator.gt_boxes = subset_boxes(full_gt, chosen)
        evaluator.pred_boxes = subset_boxes(full_pred, chosen)
        evaluator.sample_tokens = evaluator.gt_boxes.sample_tokens
        print('EVALUATING DESCRIPTION GROUP:', args.model, group, len(chosen), flush=True)
        s, coverage = save_evaluation(evaluator, out / group)
        print(json.dumps({'model': args.model, 'group': group, 'mAP': s['mean_ap'],
                          'NDS': s['nd_score'], 'absent_gt_classes': coverage['absent_gt_classes']}), flush=True)
    # Subsetting creates new containers and must not remove or reorder full boxes.
    assert full_gt.sample_tokens == full_tokens
    assert set(full_pred.sample_tokens) == tokens
    write_json(out / 'completion.json', {'status': 'WEATHER_GROUP_CPU_EVALUATION_PASS',
        'model': args.model, 'groups': list(GROUPS), 'samples': 6019,
        'inference_performed': False, 'full_replay_pass': True})


def build_report(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (work / 'preparation/scene_inventory.csv').open() as f:
        scenes = list(csv.DictReader(f))
    summaries, coverage = {}, {}
    for model in MODELS:
        assert json.loads((work / model / 'completion.json').read_text())['full_replay_pass']
        for group in GROUPS:
            summaries[model, group] = json.loads((work / model / group / 'metrics_summary.json').read_text())
            coverage[model, group] = json.loads((work / model / group / 'coverage.json').read_text())
    for g in GROUPS:
        assert coverage['lidar', g]['filtered_gt_counts'] == coverage['lidar_camera', g]['filtered_gt_counts']
    classes = list(summaries['lidar', GROUPS[0]]['mean_dist_aps'])
    common = [c for c in classes if all(coverage['lidar', g]['filtered_gt_counts'][c] > 0 for g in GROUPS)]
    rows, per_class = [], []
    for g in GROUPS:
        selected = [s for s in scenes if s['group'] == g]
        for model in MODELS:
            s = summaries[model, g]; cov = coverage[model, g]; gt = cov['filtered_gt_counts']
            represented = [c for c in classes if gt[c] > 0]
            rows.append({'group': g, 'model': model, 'scenes': len(selected),
                'samples': cov['samples'], 'logs': len({r['log_token'] for r in selected}),
                'mAP_x100': 100 * s['mean_ap'], 'NDS_x100': 100 * s['nd_score'],
                'represented_gt_classes': len(represented), 'absent_gt_classes': '|'.join(cov['absent_gt_classes']),
                'represented_classes_macro_AP_x100': 100 * sum(s['mean_dist_aps'][c] for c in represented) / len(represented),
                'common_classes_macro_AP_x100': (100 * sum(s['mean_dist_aps'][c] for c in common) / len(common)) if common else None})
            for c in classes:
                per_class.append({'group': g, 'model': model, 'class': c,
                    'filtered_GT': gt[c], 'filtered_predictions': cov['filtered_prediction_counts'][c],
                    'AP_x100': 100 * s['mean_dist_aps'][c],
                    'has_GT': gt[c] > 0, 'fewer_than_20_GT': 0 < gt[c] < 20})
    for filename, data in [('group_metrics.csv', rows), ('class_metrics.csv', per_class)]:
        with (work / filename).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    differences = {g: {key: 100 * (summaries['lidar_camera', g][key] - summaries['lidar', g][key])
                       for key in ('mean_ap', 'nd_score')} for g in GROUPS}
    write_json(work / 'paired_summary.json', {
        'status': 'PAIRED_WEATHER_GROUP_CPU_EVALUATION_PASS', 'groups': list(GROUPS),
        'common_GT_classes': common, 'paired_fusion_minus_lidar_points': differences,
        'inference_performed': False, 'full_replay_pass_both_models': True,
        'uncertainty': 'Descriptive point estimates only. No confidence intervals, p-values or significance claims. All night scenes share one acquisition log; scene resampling would not establish between-log environmental generalisation.',
        'interpretation': 'Official ten-class mAP/NDS are descriptive subgroup metrics, not official full-val benchmark scores. Absent classes retain SDK handling. Represented/common-class AP are separately labelled sensitivity summaries, not replacements for official mAP. Natural strata do not isolate weather causality or camera contribution.'})
    labels = ['Neither mentioned', 'Rain only mentioned', 'Night only mentioned', 'Night + rain mentioned']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    import numpy as np
    x = np.arange(4)
    for ax, metric, title in zip(axes, ('mean_ap', 'nd_score'), ('Ten-class mAP x100', 'NDS x100')):
        for offset, model, label, color in [(-.19, 'lidar', 'LiDAR pipeline', '#356899'),
                                          (.19, 'lidar_camera', 'Fusion pipeline', '#d67a25')]:
            values = [100 * summaries[model, g][metric] for g in GROUPS]
            bars = ax.bar(x + offset, values, .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set_xticks(x, labels, rotation=18, ha='right', fontsize=8)
        ax.set_ylim(0, 100); ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    count_text = '; '.join(f'{label}: {sum(s["group"] == g for s in scenes)} scenes / '
                          f'{len({s["log_token"] for s in scenes if s["group"] == g})} logs'
                          for label, g in zip(labels, GROUPS))
    fig.suptitle('Scene-description strata: saved full-validation predictions', fontsize=12)
    fig.text(.02, .06, count_text, fontsize=7, wrap=True)
    fig.text(.02, .025, 'No mention does not establish dry/daylight conditions. Class coverage differs; inspect class_metrics.csv. Descriptive results; no causal weather claim.', fontsize=7)
    fig.tight_layout(rect=(0, .14, 1, .94))
    for extension in ('png', 'pdf'):
        fig.savefig(work / ('weather_group_comparison.' + extension), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--base', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--report', action='store_true')
    args = p.parse_args()
    if args.report:
        build_report(args.work)
    else:
        assert args.base and args.root and args.model
        evaluate_model(args)


if __name__ == '__main__':
    main()
