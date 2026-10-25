"""Reproduce four-case diagnostics and a compact figure from preserved inputs.

Usage: python build_case_review.py --inputs /path/to/six_inputs --output .
Requires numpy, scipy, matplotlib, Pillow; no inference or GPU.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
D=a.inputs;O=a.output;O.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((D/n).read_text())
C=read('candidate_contexts.json');SD={x['filename']:x for x in read('sample_data.json')};EP={x['token']:x for x in read('ego_pose.json')};CS={x['token']:x for x in read('calibrated_sensor.json')};SR=read('sensor_records.json')
records=json.loads((O/'sources/annotation_records.json').read_text())
chosen=[(2,22,'A','Fusion-only match'),(3,26,'B','LiDAR-only match'),(5,23,'C','Shared match'),(6,7,'D','Shared miss')]
models=['lidar','lidar_camera']; colours=['#1479a8','#d56b16']
def match(c,m,threshold):
    gt=c['ground_truth'];pred=c['predictions'][m];used=set();pairs=[];fp=[]
    for pi in sorted(range(len(pred)),key=lambda i:(-pred[i]['score'],i)):
        p=pred[pi]
        if p['score']<threshold:continue
        candidates=[(float(np.linalg.norm(np.array(p['translation'][:2])-g['translation'][:2])),gi) for gi,g in enumerate(gt) if gi not in used and p['class']==g['class']]
        dist,gi=min(candidates,default=(float('inf'),-1))
        if dist<2:used.add(gi);pairs.append({'prediction_index':pi,'ground_truth_index':gi,'distance_metres':dist})
        else:fp.append(pi)
    return {'matches':pairs,'false_positive_indices':fp,'false_negative_indices':[i for i in range(len(gt)) if i not in used]}

# Recheck every original six-frame match and the saved 36 threshold summaries.
replayed={}
for t,c in C.items():
    for m in models:
        for th in [.1,.3,.5]:replayed[t,m,th]=match(c,m,th)
        r=replayed[t,m,.3];old=c['matching'][m]
        for key in ['false_positive_indices','false_negative_indices']:assert sorted(r[key])==sorted(old[key])
        pairs=lambda x:{(p['prediction_index'],p['ground_truth_index']):p['distance_metres'] for p in x['matches']}
        x,y=pairs(r),pairs(old);assert x.keys()==y.keys();assert all(abs(x[k]-y[k])<1e-10 for k in x)
for r in csv.DictReader((O/'sources/threshold_sensitivity.csv').open()):
    s=replayed[r['sample_token'],r['model'],float(r['threshold'])]
    assert [len(s[k]) for k in ['matches','false_positive_indices','false_negative_indices']]==[int(r[k]) for k in ['tp','fp','fn']]

def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def corners(b):
    w,l,h=b['size'];v=np.array([[x*l/2,y*w/2,z*h/2] for z in [-1,1] for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]]).T
    return R(b['rotation'])@v+np.array(b['translation'])[:,None]
def transform(p,pose):return R(pose['rotation']).T@(p-np.array(pose['translation'])[:,None])
def writecsv(name,rows):
    with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(9,10.3),facecolor='white');gs=fig.add_gridspec(4,2,width_ratios=[1.15,1],left=.06,right=.97,bottom=.055,top=.93,hspace=.56,wspace=.23)
objects=[];frames=[];selected=[]
for row,(case,gi,letter,title) in enumerate(chosen):
    rec=next(r for r in records if r['case']==case);t=rec['token'];c=C[t];g=c['ground_truth'][gi];obj=next(x for x in rec['highlighted_objects'] if x['gt_index']==gi)
    filename=c['sensor_files'][obj['camera']];p=D/'dataset'/filename;b=p.read_bytes();assert len(b)==SR[filename]['bytes'];assert hashlib.sha256(b).hexdigest()==SR[filename]['sha256']
    sd=SD[filename];cs=CS[sd['calibrated_sensor_token']];pts=transform(transform(corners(g),EP[sd['ego_pose_token']]),cs);assert pts[2].min()>.2
    uv=np.array(cs['camera_intrinsic'])@pts;uv=uv[:2]/uv[2];lo=uv.min(1);hi=uv.max(1)
    ax=fig.add_subplot(gs[row,0]);im=Image.open(p);ax.imshow(im)
    for j,k in [(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]:ax.plot(uv[0,[j,k]],uv[1,[j,k]],color='#ff3d92',lw=1.4)
    center=(lo+hi)/2;span=max(float(max(hi-lo))*1.7,180);width=min(span*1.65,im.width);height=min(span,im.height);cx=np.clip(center[0],width/2,im.width-width/2);cy=np.clip(center[1],height/2,im.height-height/2)
    ax.set_xlim(cx-width/2,cx+width/2);ax.set_ylim(cy+height/2,cy-height/2);ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f"{letter}. {rec['scene']} | {g['class'].replace('_',' ')}",loc='left',fontweight='bold',fontsize=11)
    ax.set_xlabel(f"{obj['camera']} | GT index {gi}",fontsize=9)
    ax=fig.add_subplot(gs[row,1]);ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.axvline(.3,color='#555',ls='--',lw=1)
    for mi,m in enumerate(models):
        pred=c['predictions'][m];dist,pi=min((float(np.linalg.norm(np.array(p['translation'][:2])-g['translation'][:2])),i) for i,p in enumerate(pred) if p['class']==g['class']);p=pred[pi]
        states={th:any(v['ground_truth_index']==gi for v in replayed[t,m,th]['matches']) for th in [.1,.3,.5]}
        assigned=next((v['prediction_index'] for v in replayed[t,m,.3]['matches'] if v['ground_truth_index']==gi),None)
        assert assigned in [None,pi] # chosen examples distinguish threshold rejection from assignment competition
        objects.append({'panel':letter,'scene':rec['scene'],'sample_token':t,'gt_index':gi,'class_name':g['class'],'camera':obj['camera'],'model':m,'nearest_prediction_index':pi,'nearest_score':p['score'],'nearest_xy_error_m':dist,'matched_at_0.1':states[.1],'matched_at_0.3':states[.3],'matched_at_0.5':states[.5]})
        y=1-mi;ax.barh(y,p['score'],height=.34,color=colours[mi],alpha=1 if states[.3] else .5)
        ax.text(.98,y+.25,f"score {p['score']:.3f} | error {dist:.3f} m",ha='right',fontsize=9)
        ax.text(.98,y-.28,'MATCH at 0.30' if states[.3] else 'MISS at 0.30',ha='right',fontsize=9,color=colours[mi])
    ax.set_yticks([1,0],['LiDAR','Fusion']);ax.set_xlim(0,1);ax.set_ylim(-.55,1.55);ax.set_xlabel('Nearest same-class candidate score');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.12)
    rr={'panel':letter,'scene':rec['scene'],'sample_token':t,'filtered_gt':len(c['ground_truth'])}
    sets=[]
    for m in models:
        z=replayed[t,m,.3];sets.append({v['ground_truth_index'] for v in z['matches']})
        for k,key in [('tp','matches'),('fp','false_positive_indices'),('fn','false_negative_indices')]:rr[m+'_'+k]=len(z[key])
    l,f=sets;rr.update(fusion_only=len(f-l),lidar_only=len(l-f),shared_matches=len(l&f),shared_misses=len(c['ground_truth'])-len(l|f));frames.append(rr)
    selected.append({'panel':letter,'original_case':case,'scene':rec['scene'],'sample_token':t,'gt_index':gi,'camera':obj['camera'],'sensor_sha256':SR[filename]['sha256']})
fig.suptitle('Four contrasting cases: scores explain the displayed match labels',fontsize=14,fontweight='bold',x=.06,ha='left',y=.985)
fig.text(.06,.955,'Ground-truth camera projections (pink) and preserved candidate scores | threshold 0.30',fontsize=10)
fig.savefig(O/'four_case_comparison.png',dpi=180);fig.savefig(O/'four_case_comparison.pdf');plt.close(fig)
writecsv('object_evidence.csv',objects);writecsv('frame_evidence.csv',frames)
(O/'selection.json').write_text(json.dumps({'selection':'Four of six previously outcome-selected cases; one previously highlighted object per frame. Editorial selection, not a representative sample.','cases':selected},indent=2)+'\n')
audit={'status':'FOUR_CASE_DIAGNOSTIC_REPLAY_PASS','original_frames_replayed':6,'model_threshold_summaries_verified':36,'selected_frames':4,'selected_objects':4,'selected_sensor_hashes_verified':4,'original_archive_sha256':'574a5ea93c4fc455317562caaa82a30f25ce3f8b20715ddbf2e05c0cff0b9bdd','scope':'Saved filtered inputs; no new inference or SDK filtering. Camera projection has no object-motion compensation or occlusion test.'}
(O/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2));print(json.dumps(frames,indent=2))
