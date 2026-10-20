"""Render one compact diagnostic figure from the saved case-analysis CSV files."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
args = parser.parse_args()
rows = list(csv.DictReader((args.data/'threshold_sensitivity.csv').open()))
objects = list(csv.DictReader((args.data/'highlighted_objects.csv').open()))
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig = plt.figure(figsize=(7.3, 6.4), layout='constrained')
grid = fig.add_gridspec(2,2,height_ratios=[1,1.1])
colours = {'lidar':'#0072B2','lidar_camera':'#D55E00'}
names = {'lidar':'LiDAR-only','lidar_camera':'Camera–LiDAR'}
for i, scene in enumerate(['scene-0332','scene-0562']):
    ax=fig.add_subplot(grid[0,i])
    for model in colours:
        selected = [r for r in rows if r['scene']==scene and r['model']==model]
        ax.plot([float(r['threshold']) for r in selected], [100*float(r['recall']) for r in selected],
                marker='o' if model=='lidar' else 's', color=colours[model], label=names[model])
    ax.set(title=f'({chr(97+i)}) {scene}', xlabel='Confidence threshold', ylabel='Matched GT (%)',
           xticks=[.1,.3,.5], ylim=(0,105))
    ax.grid(alpha=.2)
    if i==0: ax.legend(fontsize=9,loc='lower left')
ax=fig.add_subplot(grid[1,:])
labels=[]
for i,(scene,gi) in enumerate([('scene-0332',22),('scene-0332',33),('scene-0332',31),
                              ('scene-0562',26),('scene-0562',36),('scene-0562',34)]):
    selected=[r for r in objects if r['scene']==scene and int(r['gt_index'])==gi]
    scores={r['model']:float(r['nearest_same_class_score']) for r in selected}
    ax.plot([scores['lidar'], scores['lidar_camera']],[i,i],color='#aab2b8',lw=1.5)
    for model in colours:
        ax.scatter(scores[model],i,color=colours[model],marker='o' if model=='lidar' else 's',s=38,zorder=3)
    labels.append(f"{scene[6:]} · {selected[0]['class_name']} · GT {gi}")
ax.axvline(.3,color='#444444',ls='--',lw=1,label='Original threshold 0.30')
ax.set_yticks(range(6),labels=labels)
ax.invert_yaxis()
ax.set(xlim=(0,.72),xlabel='Nearest same-class prediction score (all centres <0.50 m)',
       title='(c) Highlighted outcomes depend on score eligibility')
ax.grid(axis='x',alpha=.2)
ax.legend(fontsize=9,loc='lower right')
fig.savefig(args.data/'case_diagnostics.png',dpi=220)
fig.savefig(args.data/'case_diagnostics.pdf')
plt.close(fig)
print('Composite figure saved: case_diagnostics.png and case_diagnostics.pdf')
