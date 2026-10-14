"""Plot the frozen 600-frame experiment; does not rerun accuracy evaluation."""
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
parser.add_argument('--study', type=Path, default=Path(__file__).resolve().parents[1] / 'results/nuscenes/bevfusion/camera_availability/20261014')
args = parser.parse_args()
study = args.study
evidence = study / 'original_evidence'
report = json.loads((evidence / 'evaluation/four_condition_evaluation.json').read_text())
assert report['status'] == 'FOUR_CONDITION_SUBSET_EVALUATION_PASS'
names = ['lidar_baseline', 'fusion_six_cameras', 'fusion_front_back_only', 'fusion_front_camera_missing']
labels = ['LiDAR-only', 'Fusion: six views', 'Fusion: front + back\n(four views masked)', 'Fusion: front missing\n(five views retained)']
short = ['LiDAR-only', 'Six views', 'Front + back\n(four masked)', 'Front missing\n(one masked)']
colors = ['#68778A', '#087E8B', '#D18827', '#B34263']
metrics = {}
for name in names:
    path = evidence / 'evaluation' / name / 'metrics_summary.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['conditions'][name]['metrics_summary_sha256']
    metrics[name] = json.loads(path.read_text())
    assert abs(np.mean(list(metrics[name]['mean_dist_aps'].values())) - metrics[name]['mean_ap']) < 1e-12
    nds = (5 * metrics[name]['mean_ap'] + sum(max(0., 1. - v) for v in metrics[name]['tp_errors'].values())) / 10
    assert abs(nds - metrics[name]['nd_score']) < 1e-12

figures = study / 'figures'
tables = study / 'tables'
figures.mkdir(exist_ok=True)
tables.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
def save(fig, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(figures / f'{name}.{extension}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
y = np.arange(4)
for ax, key, title in zip(axes, ['mean_ap', 'nd_score'], ['mAP (%)', 'NDS (%)']):
    vals = [100 * metrics[n][key] for n in names]
    ax.barh(y, vals, color=colors, height=.57)
    for i, v in enumerate(vals): ax.text(v + .5, i, f'{v:.2f}', va='center', fontsize=11)
    ax.set_xlim(0, 80)
    ax.set_xlabel(title)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.2)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
fig.suptitle('Camera availability: paired 600-frame nuScenes subset', weight='bold', fontsize=15)
fig.text(.5, .01, '150 scenes · Four frames per scene · Masked views remain in the six-view computation', ha='center', fontsize=10)
fig.tight_layout(rect=[0,.06,1,.94])
save(fig, 'metric_comparison')

classes = list(metrics[names[0]]['mean_dist_aps'])
counts = report['conditions'][names[0]]['filtered_gt_counts']
values = np.array([[100*metrics[n]['mean_dist_aps'][c] for n in names] for c in classes])
fig, ax = plt.subplots(figsize=(10, 7.7))
im = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(4), short)
ax.set_yticks(range(10), [c.replace('_', ' ') + f'  (GT {counts[c]:,})' for c in classes])
for i in range(10):
    for j in range(4): ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', color='white' if values[i,j]>65 else '#172A3A')
fig.colorbar(im, ax=ax, label='Per-class AP (%)', pad=.03)
ax.set_title('Per-class AP and filtered ground-truth support', pad=15, weight='bold')
fig.text(.5,.015,'Same 600 tokens in every condition · GT support is object count, not independent sample size',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.04,1,1])
save(fig, 'class_ap')

fig, ax = plt.subplots(figsize=(10.5, 6.8))
for offset, name, label, color in [(-.18,names[2],'Front + back; four masked',colors[2]),(.18,names[3],'Front missing; one masked',colors[3])]:
    vals = [100*(metrics[name]['mean_dist_aps'][c]-metrics[names[1]]['mean_dist_aps'][c]) for c in classes]
    ax.barh(np.arange(10)+offset, vals, height=.32,label=label,color=color)
ax.set_yticks(range(10), [c.replace('_',' ') for c in classes])
ax.invert_yaxis()
ax.axvline(0,color='#222',lw=1)
ax.set_xlabel('AP difference from intact six-camera fusion (percentage points)')
ax.set_title('Per-class effect of camera masking',weight='bold',pad=14)
ax.set_axisbelow(True)
ax.grid(axis='x',alpha=.2)
ax.legend(loc='lower left',frameon=False)
fig.text(.5,.015,'Same pretrained fusion weights · Inference-time normalized-image masking · No retraining',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.04,1,1])
save(fig, 'class_mask_delta')

with (tables/'comparison.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['condition','mAP','NDS','mAP_delta_vs_six_camera_points','NDS_delta_vs_six_camera_points'])
    for n in names: writer.writerow([n,metrics[n]['mean_ap'],metrics[n]['nd_score'],100*(metrics[n]['mean_ap']-metrics[names[1]]['mean_ap']),100*(metrics[n]['nd_score']-metrics[names[1]]['nd_score'])])
with (tables/'class_ap.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['class','filtered_gt_objects']+names)
    for c in classes: writer.writerow([c,counts[c]]+[metrics[n]['mean_dist_aps'][c] for n in names])
manifest={str(p.relative_to(study)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for folder in [figures,tables] for p in sorted(folder.iterdir())}
(study/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Metric arithmetic verified; generated three figures in PNG/PDF/SVG and two CSV tables.')
