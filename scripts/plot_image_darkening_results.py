"""Verify stored evidence and plot the frozen 600-frame image-darkening study."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=Path, default=Path(__file__).resolve().parents[1] / 'results/nuscenes/bevfusion/image_darkening/20261015')
study = parser.parse_args().study
e = study / 'original_evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((e / 'evidence_manifest.json').read_text())
for name, record in manifest['members'].items():
    p = e / name
    assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name
protocol = json.loads((e / 'protocol.json').read_text())
report = json.loads((e / 'evaluation/three_condition_evaluation.json').read_text())
assert report['protocol_sha256'] == sha(e / 'protocol.json')
tokens = (e / 'sample_tokens.txt').read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600
assert len(json.loads((e / 'scene_selection.json').read_text())) > 0
conditions = ['clean', 'darkening_050', 'darkening_025']
labels = ['Clean (gain 1.00)', 'Gain 0.50', 'Gain 0.25']
colors = ['#087E8B', '#C28A24', '#A93F55']
metrics = {}
for c in conditions:
    p = e / 'evaluation' / c / 'metrics_summary.json'
    m = metrics[c] = json.loads(p.read_text())
    record = report['conditions'][c]
    assert sha(p) == record['metrics_summary_sha256']
    assert record['prediction_gt_token_equality'] and record['samples'] == 600
    assert sum(record['filtered_gt_counts'].values()) == 12318
    for cls, ap in m['label_aps'].items():
        assert abs(np.mean(list(ap.values())) - m['mean_dist_aps'][cls]) < 1e-12
    assert abs(np.mean(list(m['mean_dist_aps'].values())) - m['mean_ap']) < 1e-12
    assert abs((5*m['mean_ap'] + sum(m['tp_scores'].values()))/10 - m['nd_score']) < 1e-12
    assert abs(m['mean_ap'] - record['mAP']) < 1e-12
    assert abs(m['nd_score'] - record['NDS']) < 1e-12
    for key, field in [('mAP', 'mean_ap'), ('NDS', 'nd_score')]:
        delta = 100*(m[field]-metrics['clean'][field])
        assert abs(delta-report['difference_from_fresh_clean_baseline_points'][c][key]) < 1e-12

figures = study / 'figures'
tables = study / 'tables'
figures.mkdir(exist_ok=True)
tables.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
footer = '600 frozen frames / 150 validation scenes · same pretrained fusion checkpoint · synthetic image darkening'
def save(fig, name):
    fig.text(.5, .025, footer, ha='center', fontsize=9, color='#4B5563')
    fig.tight_layout(rect=(0, .065, 1, .95))
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(figures / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, field, title in zip(axes, ['mean_ap', 'nd_score'], ['mAP', 'NDS']):
    values = [100*metrics[c][field] for c in conditions]
    ax.bar(labels, values, color=colors, width=.62)
    for i, value in enumerate(values):
        ax.text(i, value+.8, f'{value:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 80)
    ax.set_ylabel(f'{title} × 100 (higher is better)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle('Detection accuracy under controlled image darkening', fontsize=15)
save(fig, 'comparison')

classes = list(metrics['clean']['mean_dist_aps'])
y = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(10.5, 6))
class_rows = []
for k, c in enumerate(conditions[1:]):
    delta = [100*(metrics[c]['mean_dist_aps'][cls]-metrics['clean']['mean_dist_aps'][cls]) for cls in classes]
    ax.barh(y + (k-.5)*.34, delta, height=.31, color=colors[k+1], label=labels[k+1])
    for j, value in enumerate(delta):
        ax.text(value + (-.12 if value < 0 else .12), y[j]+(k-.5)*.34, f'{value:+.2f}', va='center', ha='right' if value < 0 else 'left', fontsize=8)
ax.set_yticks(y, [cls.replace('_', ' ') for cls in classes])
ax.invert_yaxis()
ax.axvline(0, color='#374151', linewidth=.9)
ax.set_xlim(-7, 1.6)
ax.set_xlabel('Class AP difference from fresh clean baseline (percentage points)')
ax.grid(axis='x', alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=2)
fig.suptitle('Class sensitivity is not uniform', fontsize=15)
save(fig, 'class_ap_delta')

errors = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
error_labels = ['Translation (m)', 'Scale (1 − IoU)', 'Orientation (rad)', 'Velocity (m/s)', 'Attribute (1 − accuracy)']
fig, axes = plt.subplots(1, 5, figsize=(14, 4.5))
for ax, key, title in zip(axes, errors, error_labels):
    values = [metrics[c]['tp_errors'][key] for c in conditions]
    ax.bar(range(3), values, color=colors, width=.65)
    ax.set_xticks(range(3), ['1.00', '0.50', '0.25'])
    ax.set_xlabel('Image gain')
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(values)*1.25)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(i, value+max(values)*.025, f'{value:.3f}', ha='center', fontsize=9)
fig.suptitle('True-positive error components (lower is better; units differ)', fontsize=14)
save(fig, 'tp_errors')

def table(name, rows):
    with (tables / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
table('headline_metrics.csv', [{'condition': c, 'image_gain': protocol['conditions'][i]['image_value_multiplier'], 'samples': 600, 'scenes': 150, 'mAP': metrics[c]['mean_ap'], 'NDS': metrics[c]['nd_score'], 'mAP_delta_points': report['difference_from_fresh_clean_baseline_points'][c]['mAP'], 'NDS_delta_points': report['difference_from_fresh_clean_baseline_points'][c]['NDS']} for i,c in enumerate(conditions)])
table('class_ap.csv', [{'class': cls, **{c: metrics[c]['mean_dist_aps'][cls] for c in conditions}, **{c+'_delta_points': 100*(metrics[c]['mean_dist_aps'][cls]-metrics['clean']['mean_dist_aps'][cls]) for c in conditions[1:]}, 'filtered_gt_count': report['conditions']['clean']['filtered_gt_counts'][cls]} for cls in classes])
table('tp_errors.csv', [{'condition': c, **metrics[c]['tp_errors']} for c in conditions])
summary = {'status': 'STORED_EVIDENCE_AND_METRIC_ARITHMETIC_PASS', 'archive_manifest_members': len(manifest['members']), 'samples': 600, 'scenes': 150, 'conditions': report['conditions'], 'differences': report['difference_from_fresh_clean_baseline_points'], 'scope': 'Compact evidence hashes and stored metric arithmetic checked; raw predictions are separate on Drive. No fresh inference, ground-truth replay or statistical significance test performed by this plotting script.'}
(study / 'analysis_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
generated = {p.relative_to(study).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for folder in (figures,tables) for p in sorted(folder.iterdir())}
(study / 'figure_manifest.json').write_text(json.dumps(generated, indent=2)+'\n')
print(json.dumps({'status': summary['status'], 'figures': 3, 'formats': ['png','pdf','svg'], 'tables': 3}, indent=2))
