"""Generate full-validation figures from the original stored nuScenes metrics."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(root):
    root = Path(root)
    report = json.loads((root / 'original_evidence/evaluation/paired_evaluation.json').read_text())
    assert report['status'] == 'PAIRED_FULL_VALIDATION_EVALUATION_PASS'
    models = report['models']; lidar = models['lidar']; fusion = models['lidar_camera']
    figdir = root / 'figures'; figdir.mkdir(exist_ok=True)
    tables = root / 'tables'; tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#355C7D', '#D27632']; labels = ['LiDAR-only', 'Camera–LiDAR']
    caption = 'Full nuScenes validation: 6,019 samples / 150 scenes • Official pretrained pipelines'
    def save(fig, name):
        fig.text(.5, .015, caption, ha='center', fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.94))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(figdir / f'{name}.{ext}', dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5,4.8)); x=np.arange(2); width=.34
    for i, model in enumerate((lidar, fusion)):
        bars=ax.bar(x+(i-.5)*width, [model['mAP']*100,model['NDS']*100],width,label=labels[i],color=colors[i])
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_xticks(x,['mAP × 100','NDS × 100']); ax.set_ylim(0,85);ax.set_ylabel('Score');ax.legend(loc='upper left')
    fig.suptitle('Full validation detection performance');save(fig,'comparison')
    classes=list(lidar['per_class_AP']); y=np.arange(len(classes))
    fig, axes=plt.subplots(1,2,figsize=(11,6),gridspec_kw={'width_ratios':[2.2,1]})
    for i,model in enumerate((lidar,fusion)):
        axes[0].barh(y+(i-.5)*.34,[100*model['per_class_AP'][c] for c in classes],.34,color=colors[i],label=labels[i])
    axes[0].set_yticks(y,[c.replace('_',' ') for c in classes]);axes[0].invert_yaxis();axes[0].set_xlim(0,100);axes[0].set_xlabel('Class AP × 100');axes[0].legend(loc='lower right')
    delta=[100*(fusion['per_class_AP'][c]-lidar['per_class_AP'][c]) for c in classes]
    bars=axes[1].barh(y,delta,color=colors[1]);axes[1].bar_label(bars,fmt='+%.2f',padding=3);axes[1].invert_yaxis();axes[1].set_yticks(y,[]);axes[1].set_xlim(0,max(delta)+2);axes[1].set_xlabel('AP point difference')
    fig.suptitle('Class detection performance and camera–LiDAR advantage');save(fig,'class_ap')
    keys=['trans_err','scale_err','orient_err','vel_err','attr_err'];names=['mATE (m)','mASE','mAOE (rad)','mAVE (m/s)','mAAE']
    fig,axes=plt.subplots(1,5,figsize=(12,4.3))
    for ax,key,name in zip(axes,keys,names):
        vals=[lidar['tp_errors'][key],fusion['tp_errors'][key]];bars=ax.bar([0,1],vals,color=colors,width=.65)
        ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8);ax.set_xticks([0,1],['L','L+C']);ax.set_ylim(0,max(vals)*1.3);ax.set_title(name)
    fig.suptitle('True-positive errors: lower is better (separate scales)');save(fig,'tp_errors')
    fig,ax=plt.subplots(figsize=(8,5.5));counts=[lidar['filtered_gt_counts'][c] for c in classes]
    bars=ax.barh(y,counts,color=colors[0]);ax.bar_label(bars,fmt='%d',padding=3,fontsize=9);ax.set_yticks(y,[c.replace('_',' ') for c in classes]);ax.invert_yaxis();ax.set_xlim(0,max(counts)*1.2);ax.set_xlabel('Filtered ground-truth boxes')
    fig.suptitle('Class support after official evaluation filtering');save(fig,'class_support')
    with (tables/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','lidar','lidar_camera','difference','units'])
        for key in ('mAP','NDS'):w.writerow([key,lidar[key]*100,fusion[key]*100,(fusion[key]-lidar[key])*100,'score points'])
        for key in keys:w.writerow([key,lidar['tp_errors'][key],fusion['tp_errors'][key],fusion['tp_errors'][key]-lidar['tp_errors'][key],'native metric units'])
    with (tables/'class_ap.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['class','lidar_AP_x100','lidar_camera_AP_x100','difference_points','filtered_GT_boxes'])
        for c,d in zip(classes,delta):w.writerow([c,lidar['per_class_AP'][c]*100,fusion['per_class_AP'][c]*100,d,lidar['filtered_gt_counts'][c]])
    manifest={'source':'original_evidence/evaluation/paired_evaluation.json','source_sha256':hashlib.sha256((root/'original_evidence/evaluation/paired_evaluation.json').read_bytes()).hexdigest(),'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figdir.iterdir())},'scope':'Descriptive single evaluation; no confidence intervals or significance claims.'}
    (root/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/nuscenes/bevfusion/full_validation')
