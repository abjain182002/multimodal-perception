"""Verify stored measurements and plot the frozen 150-frame runtime study."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--study',type=Path,default=Path(__file__).resolve().parents[1]/'results/nuscenes/bevfusion/runtime/20261015')
study=parser.parse_args().study
e=study/'original_evidence'; report=json.loads((e/'benchmark_audit.json').read_text())
tokens=(e/'sample_tokens.txt').read_text().splitlines()
assert len(tokens)==len(set(tokens))==150
models=['lidar','lidar_camera']; labels=['LiDAR-only','Camera + LiDAR']; colors=['#68778A','#087E8B']
rows={}; summaries={}; environments={}
for s in report['sessions']:
    folder=e/'sessions'/s['session']; path=folder/'measurements.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==s['measurements_sha256']
    raw=json.loads(path.read_text()); assert [r['sample_token'] for r in raw]==tokens
    x=np.array([r['test_step_latency_ms'] for r in raw]); l=np.array([r['loading_and_collation_ms'] for r in raw])
    for k,v in {'mean':np.mean(x),'median':np.median(x),'p95':np.percentile(x,95),'minimum':x.min(),'maximum':x.max()}.items(): assert abs(s['latency_ms'][k]-v)<1e-9
    for k,v in {'mean':l.mean(),'median':np.median(l),'p95':np.percentile(l,95)}.items(): assert abs(s['loading_and_collation_ms'][k]-v)<1e-9
    rows[(s['repeat'],s['model'])]=raw; summaries[(s['repeat'],s['model'])]=s
    environments[s['session']]=json.loads((folder/'environment.json').read_text())
for repeat in (1,2,3):
    assert [r['input_points'] for r in rows[(repeat,'lidar')]]==[r['input_points'] for r in rows[(repeat,'lidar_camera')]]
figures=study/'figures'; tables=study/'tables';figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
aggregate={}
for m in models:
    ss=[summaries[(r,m)] for r in (1,2,3)]
    pooled=np.array([v['test_step_latency_ms'] for r in (1,2,3) for v in rows[(r,m)]])
    loads=np.array([v['loading_and_collation_ms'] for r in (1,2,3) for v in rows[(r,m)]])
    aggregate[m]={'median_of_session_medians_ms':float(np.median([s['latency_ms']['median'] for s in ss])), 'pooled_mean_ms':float(pooled.mean()),'pooled_median_ms':float(np.median(pooled)),'pooled_p95_ms':float(np.percentile(pooled,95)), 'session_medians_ms':[s['latency_ms']['median'] for s in ss],'session_p95_ms':[s['latency_ms']['p95'] for s in ss], 'pooled_loading_mean_ms':float(loads.mean()), 'max_peak_allocated_MiB':max(s['memory_bytes']['peak_allocated'] for s in ss)/2**20,'max_peak_reserved_MiB':max(s['memory_bytes']['peak_reserved'] for s in ss)/2**20}

fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,key,title in zip(axes,['median','p95'],['Session median','Session 95th percentile']):
    for j,m in enumerate(models):
        vals=[summaries[(r,m)]['latency_ms'][key] for r in (1,2,3)]
        ax.plot([1,2,3],vals,'o-',color=colors[j],label=labels[j],lw=2)
        for i,v in enumerate(vals):ax.annotate(f'{v:.1f}',(i+1,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks([1,2,3]);ax.set_xlabel('Repeat');ax.set_ylabel('Latency (ms)');ax.set_ylim(0,680);ax.set_title(title);ax.grid(alpha=.2)
axes[0].legend(frameon=False,loc='lower left')
fig.suptitle('Serial inference latency: same 150 samples on Tesla T4',weight='bold',fontsize=14)
fig.text(.5,.01,'Batch size 1 · 20 warm-up samples/session · Three isolated sessions/model · Dataset loading excluded',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93]);save(fig,'latency_sessions')

fig,axes=plt.subplots(1,2,figsize=(10,4.8))
for ax,key,title in zip(axes,['max_peak_allocated_MiB','max_peak_reserved_MiB'],['Peak allocated','Peak reserved']):
    vals=[aggregate[m][key] for m in models];ax.bar(labels,vals,color=colors,width=.55)
    for i,v in enumerate(vals):ax.text(i,v+45,f'{v:,.1f}',ha='center')
    ax.set_ylim(0,3700);ax.set_ylabel('GPU memory (MiB)');ax.set_title(title);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
fig.suptitle('PyTorch GPU memory: maximum across three sessions',weight='bold',fontsize=14)
fig.text(.5,.01,'Includes weights · Peaks reset after warm-up · Reserved includes cached blocks · Not total device memory',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93]);save(fig,'gpu_memory')

accuracy=json.loads((study/'accuracy_context/evaluation/four_condition_evaluation.json').read_text())['conditions']
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,metric,title in zip(axes,['mAP','NDS'],['mAP (%)','NDS (%)']):
    for j,m in enumerate(models):
        a=aggregate[m];v=accuracy['lidar_baseline' if m=='lidar' else 'fusion_six_cameras'][metric]*100
        mid=a['median_of_session_medians_ms'];vs=a['session_medians_ms']
        ax.errorbar(mid,v,xerr=[[mid-min(vs)],[max(vs)-mid]],fmt='o',ms=9,color=colors[j],capsize=5)
        ax.annotate(labels[j]+f'\n{v:.2f}',(mid,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xlim(200,640);ax.set_ylim(60 if metric=='mAP' else 66,74 if metric=='mAP' else 76);ax.set_xlabel('Median of session medians (ms)');ax.set_ylabel(title);ax.grid(alpha=.2)
fig.suptitle('Accuracy and compute: two separately measured scopes',weight='bold',fontsize=14)
fig.text(.5,.01,'Accuracy: 600 frames · Runtime: nested 150 frames · Horizontal range: session medians, not confidence intervals',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93]);save(fig,'accuracy_latency')

paired=[]
for repeat in (1,2,3):
    for a,b in zip(rows[(repeat,'lidar')],rows[(repeat,'lidar_camera')]):paired.append({'repeat':repeat,'sample_token':a['sample_token'],'input_points':a['input_points'],'lidar_ms':a['test_step_latency_ms'],'fusion_ms':b['test_step_latency_ms'],'fusion_over_lidar':b['test_step_latency_ms']/a['test_step_latency_ms']})
ratios=np.array([p['fusion_over_lidar'] for p in paired])
summary={'status':'RUNTIME_ANALYSIS_PASS','samples':150,'sessions':6,'timed_predictions':900,'models':aggregate,'ratio_of_median_session_medians':aggregate['lidar_camera']['median_of_session_medians_ms']/aggregate['lidar']['median_of_session_medians_ms'],'peak_allocated_ratio':aggregate['lidar_camera']['max_peak_allocated_MiB']/aggregate['lidar']['max_peak_allocated_MiB'],'paired_latency_ratios':{'median':float(np.median(ratios)),'p05':float(np.percentile(ratios,5)),'p95':float(np.percentile(ratios,95))},'accuracy_scope':'600-frame subset, not newly evaluated on 150','limitations':['Shared Colab runtime; clocks not locked.','End-session environment snapshots are not continuous telemetry.','Three repeats are not 450 independent scenes.','Timing excludes file loading and CPU validation.','Memory is PyTorch allocated/reserved, not total device memory.','Runtime TF32 disabled; accuracy context was measured separately.']}
(study/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
def table(name,records):
    with (tables/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
table('paired_latency.csv',paired)
table('session_summary.csv',[{'model':s['model'],'repeat':s['repeat'],'mean_ms':s['latency_ms']['mean'],'median_ms':s['latency_ms']['median'],'p95_ms':s['latency_ms']['p95'],'loading_mean_ms':s['loading_and_collation_ms']['mean'],'peak_allocated_MiB':s['memory_bytes']['peak_allocated']/2**20,'peak_reserved_MiB':s['memory_bytes']['peak_reserved']/2**20} for s in report['sessions']])
table('accuracy_compute_matrix.csv',[{'model':m,'accuracy_samples':600,'runtime_samples':150,'runtime_repeats':3,'mAP':accuracy['lidar_baseline' if m=='lidar' else 'fusion_six_cameras']['mAP'],'NDS':accuracy['lidar_baseline' if m=='lidar' else 'fusion_six_cameras']['NDS'],**{k:v for k,v in aggregate[m].items() if not isinstance(v,list)}} for m in models])
manifest={p.relative_to(study).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for folder in [figures,tables] for p in sorted(folder.iterdir())}
(study/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
