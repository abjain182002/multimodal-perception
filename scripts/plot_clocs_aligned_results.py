"""Generate exact CSV tables and dependency-free SVG report figures."""
import csv
import hashlib
import html
import json
from pathlib import Path


def main():
    root = Path('results/clocs_aligned_20260930')
    ap = json.loads((root / 'ap40.json').read_text())
    seeds = json.loads((root / 'seed_comparison.json').read_text())
    assert ap['status'] == seeds['status'] == 'PASS'
    assert ap['frames'] == 3769 and ap['baseline_max_absolute_ap40_error'] == 0
    pv = json.loads(Path('results/pv_rcnn_kitti/common_ap40.json').read_text())
    assert pv['status'] == 'PASS' and pv['maximum_absolute_ap_difference'] == 0
    methods = [('PointPillars', ap['ap40_strict_iou_070']['pointpillars']),
               ('Aligned CLOCs', ap['ap40_strict_iou_070']['fusion']),
               ('Official PV-RCNN', pv['ap40_strict_iou_070'])]
    colors = ['#416a92', '#d17836', '#448371']
    def save_svg(name, title, body, footer):
        content = ('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560">'
                   '<rect width="960" height="560" fill="white"/>'
                   '<g font-family="Arial,sans-serif" fill="#222">'
                   f'<text x="480" y="32" text-anchor="middle" font-size="20">{html.escape(title)}</text>'
                   + body + f'<text x="480" y="540" text-anchor="middle" font-size="12">{html.escape(footer)}</text></g></svg>')
        (root / name).write_text(content + '\n')
    body = ''
    for tick in range(0, 101, 20):
        y = 450 - tick * 3.6
        body += f'<path d="M80 {y}H910" stroke="#ddd"/><text x="65" y="{y+4}" text-anchor="end">{tick}</text>'
    rows = []
    for i, difficulty in enumerate(('easy', 'moderate', 'hard')):
        center = 230 + i * 270
        body += f'<text x="{center}" y="480" text-anchor="middle">{difficulty.title()}</text>'
        for j, (name, values) in enumerate(methods):
            value = values['3D'][difficulty]
            x = center - 100 + j * 68
            body += f'<rect x="{x}" y="{450-value*3.6}" width="58" height="{value*3.6}" fill="{colors[j]}"/>'
            body += f'<text x="{x+29}" y="{440-value*3.6}" text-anchor="middle" font-size="13">{value:.2f}</text>'
        for metric in ('2D', 'BEV', '3D'):
            base = methods[0][1][metric][difficulty]
            fusion = methods[1][1][metric][difficulty]
            rows.append([metric, difficulty, base, fusion, fusion-base, methods[2][1][metric][difficulty]])
    for j, (name, _) in enumerate(methods):
        x = 100+j*285
        body += f'<rect x="{x}" y="505" width="16" height="12" fill="{colors[j]}"/><text x="{x+24}" y="516" font-size="14">{name}</text>'
    save_svg('three_model_3d_ap40.svg', 'KITTI validation: strict Car 3D AP40 (IoU 0.70)', body,
             '3,769 frames; common evaluator. Complete pretrained pipelines; training budgets differ.')
    with (root / 'ap40_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['metric','difficulty','pointpillars','aligned_clocs','fusion_minus_baseline_ap_points','pv_rcnn'])
        writer.writerows(rows)
    body = ''
    for tick in (0, .2, .4, .6, .8, 1):
        y = 450-tick*360
        body += f'<path d="M80 {y}H910" stroke="#ddd"/><text x="65" y="{y+4}" text-anchor="end">{tick:.1f}</text>'
    base = seeds['runs'][0]['pointpillars_candidate_ap']
    body += f'<path d="M80 {450-base*360}H910" stroke="#555" stroke-dasharray="6 4"/>'
    curve_rows = []
    for j, run in enumerate(seeds['runs']):
        history_path = root / f"seed_{run['seed']}/history.json"
        history = json.loads(history_path.read_text())
        assert len(history) == 12
        points = []
        for record in history:
            epoch = record['epoch']
            value = record['dev']['fusion_candidate_ap']
            points.append(f'{90+(epoch-1)*73},{450-value*360}')
            curve_rows.append([run['seed'], epoch, value, record['mean_training_selected_loss']])
        body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[j]}" stroke-width="2.5"/>'
        x = 100+j*260
        body += f'<path d="M{x} 505h20" stroke="{colors[j]}" stroke-width="3"/><text x="{x+28}" y="510">Seed {run["seed"]}</text>'
    for epoch in range(1,13):
        body += f'<text x="{90+(epoch-1)*73}" y="478" text-anchor="middle">{epoch}</text>'
    body += '<text x="480" y="493" text-anchor="middle" font-size="12">Epoch</text>'
    save_svg('seed_learning_curves.svg', 'Internal development candidate AP: three fixed-split seeds', body,
             '96 optimization / 32 development frames; dashed line: PointPillars candidate AP. Not KITTI AP40.')
    with (root / 'seed_learning_curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['seed','epoch','dev_candidate_ap','selected_train_loss'])
        writer.writerows(curve_rows)
    findings = '''# Aligned CLOCs findings — 30 September 2026

FOV-reduced training candidates were regenerated for the same 128 KITTI training frames. The original 96 optimization / 32 internal development split was fixed across three training seeds. Each run used all valid candidates and 12 epochs. Mean best development candidate AP was 0.870549, sample standard deviation 0.006397; PointPillars candidate AP was 0.777660. Every run selected epoch 12, so convergence remains unverified.

Seed 20260928 was selected by internal development candidate AP before this validation run. Full validation used this selected seed only; the three-seed statistics are not full-validation uncertainty estimates.

On 3,769 validation frames, the frozen PointPillars baseline reproduced all nine reference AP40 values exactly (maximum error 0.0). Aligned CLOCs 3D AP40 was 90.7684 Easy, 79.8616 Moderate and 74.8719 Hard. Relative to PointPillars: +0.4235, +0.7317 and -1.2846 AP points respectively. The mixed outcome must be reported, including the Hard decline.

Official PV-RCNN Moderate 3D AP40 was 84.3604 under the common evaluator, 4.4989 points above this aligned CLOCs run. This compares complete pipelines with different architectures and training budgets; it does not establish a general limitation of fusion. PV-RCNN is a two-stage detector, not PV-RCNN++ or Azitt.

Only 96 frames optimized the fusion head. Seed repeats do not answer the training-size question. Validation had already been inspected in earlier experiments, so these results are an iterative validation comparison rather than an untouched final test. The observed improvement over the earlier checkpoint does not isolate preprocessing as a causal effect because training environment and checkpoint selection also differ.

Sources: ap40.json, seed_comparison.json, study_protocol.json, validation_selection.json and each seed's history.json; PV-RCNN: ../pv_rcnn_kitti/common_ap40.json. SVG figures preserve editable vector output; CSV tables retain unrounded values.
'''
    (root / 'FINDINGS.md').write_text(findings)
    generated = ['three_model_3d_ap40.svg','seed_learning_curves.svg','ap40_comparison.csv','seed_learning_curves.csv','FINDINGS.md']
    manifest = {'sources': {}, 'outputs': {}}
    for path in [root/'ap40.json',root/'seed_comparison.json',Path('results/pv_rcnn_kitti/common_ap40.json')]:
        manifest['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in generated:
        manifest['outputs'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    (root/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('FIGURES AND FINDINGS PASS:', root)


if __name__ == '__main__':
    main()
