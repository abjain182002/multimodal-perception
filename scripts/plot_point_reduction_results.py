"""Verify stored point-reduction evidence and generate compact report figures."""
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
parser.add_argument('--study', type=Path, default=Path(__file__).resolve().parents[1] / 'results/nuscenes/bevfusion/point_reduction/20261015')
study = parser.parse_args().study
e = study / 'original_evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((e / 'evidence_manifest.json').read_text())
for name, record in manifest['members'].items():
    p = e / name
    assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name
report = json.loads((e / 'evaluation/six_condition_evaluation.json').read_text())
inference = json.loads((e / 'inference/six_condition_inference.json').read_text())
protocol = json.loads((e / 'protocol.json').read_text())
assert sha(e / 'protocol.json') == report['protocol_sha256']
tokens = (e / 'sample_tokens.txt').read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600
selection = json.loads((e / 'selected_input_manifest.json').read_text())
assert len(selection) == 600 and len({r['scene_token'] for r in selection}) == 150
models = ['lidar', 'lidar_camera']
suffixes = ['full', 'keep050', 'keep025']
fractions = [1., .5, .25]
metrics, inputs = {}, {}
for record in inference['conditions']:
    c = record['condition']
    folder = e / 'inference' / c
    assert sha(folder / 'protocol.json') == record['protocol_sha256']
    assert sha(folder / 'input_checks.json') == record['input_checks_sha256']
    ledger = json.loads((folder / 'input_checks.json').read_text())
    assert ledger['protocol_sha256'] == record['protocol_sha256']
    inputs[c] = ledger['records']
    assert [r['sample_token'] for r in inputs[c]] == tokens
    p = e / 'evaluation' / c / 'metrics_summary.json'
    m = metrics[c] = json.loads(p.read_text())
    audit = report['conditions'][c]
    assert sha(p) == audit['metrics_summary_sha256']
    assert record['results_sha256'] == audit['predictions_sha256']
    assert audit['prediction_gt_token_equality'] and sum(audit['filtered_gt_counts'].values()) == 12318
    for cls, ap in m['label_aps'].items():
        assert abs(np.mean(list(ap.values())) - m['mean_dist_aps'][cls]) < 1e-12
    assert abs(np.mean(list(m['mean_dist_aps'].values())) - m['mean_ap']) < 1e-12
    assert abs((5*m['mean_ap']+sum(m['tp_scores'].values()))/10-m['nd_score']) < 1e-12
    assert abs(m['mean_ap']-audit['mAP']) < 1e-12
    assert abs(m['nd_score']-audit['NDS']) < 1e-12

for index, token in enumerate(tokens):
    rows = {c: records[index] for c,records in inputs.items()}
    assert len({(r['original_points'], r['original_points_sha256']) for r in rows.values()}) == 1
    n = rows['lidar_full']['original_points']
    seed = int.from_bytes(hashlib.sha256(('20261002:'+token).encode()).digest()[:8], 'big')
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    for suffix, fraction in zip(suffixes, fractions):
        indices = np.arange(n, dtype=np.int64) if fraction == 1 else np.sort(permutation[:max(1, int(np.floor(fraction*n)))])
        digest = hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
        a, b = rows['lidar_'+suffix], rows['lidar_camera_'+suffix]
        assert a['retained_points'] == b['retained_points'] == len(indices)
        assert a['selected_indices_sha256'] == b['selected_indices_sha256'] == digest
        assert a['selected_points_sha256'] == b['selected_points_sha256']
    for model in models:
        assert len({(rows[model+'_'+s]['images_sha256'], rows[model+'_'+s]['metadata_sha256']) for s in suffixes}) == 1

for model in models:
    baseline = metrics[model+'_full']
    for suffix in suffixes:
        c = model+'_'+suffix
        for key,field in [('mAP','mean_ap'), ('NDS','nd_score')]:
            assert abs(100*(metrics[c][field]-baseline[field])-report['difference_from_own_full_baseline_points'][c][key]) < 1e-12
for suffix in suffixes:
    for key,field in [('mAP','mean_ap'), ('NDS','nd_score')]:
        assert abs(100*(metrics['lidar_camera_'+suffix][field]-metrics['lidar_'+suffix][field])-report['fusion_minus_lidar_at_same_retention_points'][suffix][key]) < 1e-12

figures, tables = study/'figures', study/'tables'
figures.mkdir(exist_ok=True)
tables.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
colors = ['#68778A','#087E8B']
labels = ['LiDAR-only','Camera + LiDAR']
footer = '600 frozen frames / 150 scenes · identical retained points across models · synthetic thinning, not a physical LiDAR model'
def save(fig,name):
    fig.text(.5,.02,footer,ha='center',fontsize=9,color='#4B5563')
    fig.tight_layout(rect=(0,.065,1,.94))
    for ext in ('png','pdf','svg'):
        fig.savefig(figures/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(11,4.8))
for ax,field,title in zip(axes,['mean_ap','nd_score'],['(a) Detection mAP','(b) nuScenes Detection Score']):
    for model,label,color in zip(models,labels,colors):
        values = [100*metrics[model+'_'+s][field] for s in suffixes]
        ax.plot([100,50,25],values,'o-',color=color,label=label,linewidth=2,markersize=6)
        for x,y in zip([100,50,25],values):
            ax.annotate(f'{y:.2f}',(x,y),xytext=(0,8 if model=='lidar_camera' else -15),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks([100,50,25])
    ax.set_xlim(110,15)
    ax.set_ylim(45,76)
    ax.set_xlabel('LiDAR points retained (%)')
    ax.set_ylabel(('mAP' if field=='mean_ap' else 'NDS')+' × 100 (higher is better)')
    ax.set_title(title)
    ax.grid(alpha=.2)
    ax.legend(loc='lower left',fontsize=9)
fig.suptitle('Accuracy under matched LiDAR point reduction',fontsize=15)
save(fig,'comparison')

classes = list(metrics['lidar_full']['mean_dist_aps'])
fig,axes=plt.subplots(1,2,figsize=(11,6.4),sharey=True)
for ax,model,label in zip(axes,models,labels):
    values=np.array([[100*(metrics[model+'_'+s]['mean_dist_aps'][cls]-metrics[model+'_full']['mean_dist_aps'][cls]) for s in suffixes[1:]] for cls in classes])
    limit=max(1,max(abs(100*(metrics[m+'_'+s]['mean_dist_aps'][cls]-metrics[m+'_full']['mean_dist_aps'][cls])) for m in models for s in suffixes[1:] for cls in classes))
    im=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks([0,1],['50% retained','25% retained'])
    ax.set_yticks(range(len(classes)),[c.replace('_',' ') for c in classes])
    ax.set_title(label)
    for j in range(len(classes)):
        for k in range(2):
            ax.text(k,j,f'{values[j,k]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(values[j,k])>limit*.6 else '#111827')
fig.suptitle('Class AP change from each model’s own full-point baseline (points)',fontsize=13)
save(fig,'class_ap_delta')

def table(name,rows):
    with (tables/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
table('headline_metrics.csv',[{'model':m,'condition':m+'_'+s,'retained_fraction':f,'mAP':metrics[m+'_'+s]['mean_ap'],'NDS':metrics[m+'_'+s]['nd_score'],'mAP_delta_points':report['difference_from_own_full_baseline_points'][m+'_'+s]['mAP'],'NDS_delta_points':report['difference_from_own_full_baseline_points'][m+'_'+s]['NDS']} for m in models for s,f in zip(suffixes,fractions)])
table('class_ap.csv',[{'model':m,'class':cls,'full_AP':metrics[m+'_full']['mean_dist_aps'][cls],**{s+'_AP':metrics[m+'_'+s]['mean_dist_aps'][cls] for s in suffixes[1:]},**{s+'_delta_points':100*(metrics[m+'_'+s]['mean_dist_aps'][cls]-metrics[m+'_full']['mean_dist_aps'][cls]) for s in suffixes[1:]},'filtered_gt_count':report['conditions'][m+'_full']['filtered_gt_counts'][cls]} for m in models for cls in classes])
table('point_counts.csv',[{'condition':c,'samples':600,'original_mean':float(np.mean([r['original_points'] for r in records])),'retained_min':min(r['retained_points'] for r in records),'retained_mean':float(np.mean([r['retained_points'] for r in records])),'retained_max':max(r['retained_points'] for r in records)} for c,records in inputs.items()])
table('fusion_advantage.csv',[{'retained_fraction':f,**report['fusion_minus_lidar_at_same_retention_points'][s]} for s,f in zip(suffixes,fractions)])
summary={'status':'POINT_REDUCTION_STORED_EVIDENCE_ANALYSIS_PASS','samples':600,'scenes':150,'manifest_listed_members':len(manifest['members']),'all_600_input_pairings_verified':True,'all_600_index_selection_hashes_recomputed':True,'own_baseline_differences':report['difference_from_own_full_baseline_points'],'fusion_advantage':report['fusion_minus_lidar_at_same_retention_points'],'scope':'Compact member integrity, stored metric arithmetic and per-frame input records checked. Raw sensor tensors/predictions are separate on Drive; no fresh inference, ground-truth replay or significance test.'}
(study/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
generated={p.relative_to(study).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for folder in (figures,tables) for p in sorted(folder.iterdir())}
(study/'figure_manifest.json').write_text(json.dumps(generated,indent=2)+'\n')
print(json.dumps(summary,indent=2))
