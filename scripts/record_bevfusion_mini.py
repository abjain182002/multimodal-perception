"""Verify and record the 81-sample nuScenes pretrained BEVFusion pilot."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

EXPECTED = '03de2aea2c58045671b7462ed5f44f4e246efedb69ad1469ca28f070a4b3a909'
PREFIX = 'bevfusion_mini_validation/'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    digest = sha256(args.archive)
    if digest != EXPECTED:
        raise ValueError(f'Archive hash mismatch: {digest}')
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/bevfusion_nuscenes_mini_20261001'

    with zipfile.ZipFile(args.archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP entries'
        manifest = json.loads(bundle.read('evidence_manifest.json'))
        members = manifest['members']
        assert set(names) == set(members) | {'evidence_manifest.json'}
        for name, record in members.items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            content = bundle.read(name)
            assert len(content) == record['bytes'], name
            assert hashlib.sha256(content).hexdigest() == record['sha256'], name

        def read_json(name):
            return json.loads(bundle.read(name))

        paired = read_json(PREFIX + 'paired_evaluation.json')
        assert paired['status'] == 'PAIRED_MINI_VALIDATION_PASS'
        assert paired['same_ordered_sample_tokens'] is True
        tokens = paired['sample_tokens']
        assert len(tokens) == len(set(tokens)) == 81
        summaries = {}
        for model in ('lidar', 'lidar_camera'):
            audit = paired['models'][model]
            assert audit['status'] == 'MINI_VALIDATION_PASS'
            assert audit['frames'] == 81 and audit['eval_split'] == 'mini_val'
            assert audit['version'] == 'v1.0-mini'
            pred_prefix = PREFIX + model + '/predictions/'
            pred_names = sorted(n for n in names if n.startswith(pred_prefix) and n.endswith('.json'))
            assert len(pred_names) == 81
            assert [read_json(n)['sample_token'] for n in pred_names] == tokens
            official = PREFIX + model + '/official_results/pred_instances_3d/'
            predictions = read_json(official + 'results_nusc.json')
            assert set(predictions['results']) == set(tokens)
            summary = read_json(official + 'metrics_summary.json')
            for key, raw_key in (('/mAP', 'mean_ap'), ('/NDS', 'nd_score')):
                values = [v for k, v in audit['metrics'].items() if k.endswith(key)]
                assert len(values) == 1 and abs(values[0] - summary[raw_key]) < 1e-12
            summaries[model] = summary

        # All validation precedes writing repository reports.
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            keep = name in ('evidence_manifest.json', PREFIX + 'paired_evaluation.json')
            keep |= name.startswith('setup/') or name.startswith('source/')
            keep |= name.endswith(('evaluation_audit.json', 'metrics_summary.json', 'metrics_details.json'))
            keep |= name in ('metadata/metadata_audit.json', PREFIX + 'evaluate.py', PREFIX + 'evaluation.log')
            keep |= name.startswith('bevfusion_mini_smoke/')
            if keep:
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(name))

    receipt = dict(status='ARCHIVE_AND_MEMBERS_VERIFIED', archive_sha256=digest,
                   bytes=args.archive.stat().st_size, members_verified=len(members),
                   prediction_files_verified=162, frames=81)
    (output / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    rows = []
    for label, key in (('mAP', 'mean_ap'), ('NDS', 'nd_score')):
        lidar = summaries['lidar'][key] * 100
        fusion = summaries['lidar_camera'][key] * 100
        rows.append((label, lidar, fusion, fusion - lidar))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['metric_times_100', 'lidar', 'lidar_camera', 'difference_points'])
        writer.writerows(rows)
    with (output / 'per_class_ap.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['class', 'lidar_ap_times_100', 'lidar_camera_ap_times_100', 'difference_points'])
        for name, value in summaries['lidar']['mean_dist_aps'].items():
            other = summaries['lidar_camera']['mean_dist_aps'][name]
            writer.writerow([name, value * 100, other * 100, (other - value) * 100])

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" viewBox="0 0 760 440">',
           '<rect width="760" height="440" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#202020">',
           '<text x="45" y="30" font-size="20">nuScenes mini_val: official pretrained checkpoints</text>',
           '<text x="45" y="54" font-size="13">81 samples / 2 scenes; mini pilot, not full validation</text>']
    for tick in range(0, 101, 20):
        y = 350 - tick * 2.5
        svg.append(f'<line x1="80" y1="{y}" x2="700" y2="{y}" stroke="#dddddd"/>')
        svg.append(f'<text x="42" y="{y + 5}" font-size="12">{tick}</text>')
    for index, (label, lidar, fusion, delta) in enumerate(rows):
        x = 190 + index * 280
        for offset, value, color in ((0, lidar, '#4477aa'), (85, fusion, '#228833')):
            height = value * 2.5
            svg.append(f'<rect x="{x + offset}" y="{350 - height}" width="65" height="{height}" fill="{color}"/>')
            svg.append(f'<text x="{x + offset + 32}" y="{340 - height}" text-anchor="middle" font-size="13">{value:.2f}</text>')
        svg.append(f'<text x="{x + 75}" y="375" text-anchor="middle" font-size="14">{label} × 100</text>')
    svg.extend(['<text x="190" y="410" fill="#4477aa" font-size="14">LiDAR only</text>',
                '<text x="420" y="410" fill="#228833" font-size="14">Camera–LiDAR</text>', '</g></svg>'])
    (output / 'comparison.svg').write_text('\n'.join(svg) + '\n')
    findings = f'''# nuScenes mini pretrained BEVFusion pilot

Both official OpenMMLab checkpoints passed strict loading and were evaluated
on the same ordered 81 mini_val samples from two scenes using the pinned
MMDetection3D NuScenesMetric and nuScenes evaluator.

Camera–LiDAR minus LiDAR-only: mAP {rows[0][3]:+.4f} points and
NDS {rows[1][3]:+.4f} points (metrics multiplied by 100).

This compares two pretrained checkpoint pipelines. It is not a controlled
camera ablation, training experiment, or full nuScenes validation reproduction.
Trailer, construction vehicle and barrier have no annotated instances in this
mini_val subset. Preserve official absent-class and undefined-error outputs;
do not interpret them as observed class failures. Class counts printed by the
dataset are not necessarily the evaluator's post-filter ground-truth counts.
The dataset version was explicitly configured as v1.0-mini after the default
v1.0-trainval caused a pre-inference assertion failure. Prediction settings and
official evaluation class definitions were retained.

Figure: comparison.svg shows official mini_val mAP and NDS multiplied by 100,
with an axis from 0 to 100. Only these two scenes support this pilot comparison.
Checkpoint identities, source files, per-class metrics and setup logs are saved
with the archive receipt. The full ZIP retains all 162 per-sample predictions
and both official results_nusc.json files outside Git.
'''
    (output / 'FINDINGS.md').write_text(findings)
    artifacts = ['comparison.csv', 'per_class_ap.csv', 'comparison.svg', 'FINDINGS.md']
    (output / 'figure_manifest.json').write_text(json.dumps({
        'script_sha256': sha256(Path(__file__)),
        'artifacts': {name: sha256(output / name) for name in artifacts},
    }, indent=2) + '\n')
    print('MINI EVIDENCE, COMPARISON AND FIGURE SAVED:', output)


if __name__ == '__main__':
    main()
