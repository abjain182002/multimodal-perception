"""CPU annotation-based mobile-robot alert analysis; no model inference."""
import argparse,csv,hashlib,itertools,json,pickle,platform,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(b): return hashlib.sha256(b).hexdigest()
def rotation(q):
    q=np.asarray(q,float); assert abs(np.linalg.norm(q)-1)<1e-5
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def hull(center,dims,rot,transform):
    corners=np.array(list(itertools.product(*[[-v/2,v/2] for v in dims])))@rot.T+center
    corners=np.c_[corners,np.ones(8)]@transform.T
    xy=corners[:,:2]; return xy[ConvexHull(xy).vertices]
def intersects(a,b):
    if np.any(a.max(0)<b.min(0)-1e-10) or np.any(b.max(0)<a.min(0)-1e-10): return False
    for p in (a,b):
        for edge in np.roll(p,-1,axis=0)-p:
            v=np.array([-edge[1],edge[0]]); aa=a@v; bb=b@v
            if aa.max()<bb.min()-1e-10 or bb.max()<aa.min()-1e-10:return False
    return True
def writer(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(n,d): return float(n/d) if d else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--output',default='results');a=ap.parse_args()
    root=Path(__file__).resolve().parent; pr=json.loads((root/'protocol.json').read_text()); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    raw=Path(a.inputs).read_bytes();assert sha(raw)==pr['input_zip_sha256']; z=zipfile.ZipFile(a.inputs)
    manifest=json.loads(z.read('manifest.json'))
    for n,h in manifest.items(): assert sha(z.read(n))==h,n
    tokens=z.read('sample_tokens.txt').decode().splitlines();assert len(tokens)==len(set(tokens))==600
    ann=pickle.loads(z.read('annotations.pkl'));data=ann['data_list'];assert [d['token'] for d in data]==tokens
    cats={v:k for k,v in ann['metainfo']['categories'].items()}
    pred={m:json.loads(z.read(n))['results'] for m,n in [('lidar','lidar_predictions.json'),('fusion','fusion_predictions.json')]}
    assert all(set(p)==set(tokens) for p in pred.values())
    scenes=json.loads((root/'scene_selection.json').read_text());assert len(scenes)==150
    lookup={t:(s['scene_token'],s['scene_name']) for s in scenes for t in s['sample_tokens']};assert set(lookup)==set(tokens)
    assert all(len(s['sample_tokens'])==4 for s in scenes)
    ranges={c:(30 if c in ['barrier','traffic_cone'] else 40 if c in ['pedestrian','bicycle','motorcycle'] else 50) for c in cats.values()}
    sq=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    assert intersects(sq,sq+[1,0]) and not intersects(sq,sq+[1.01,0]) and intersects(sq,sq*.2+.3)
    assert np.allclose(rotation([2**-.5,0,0,2**-.5])@[1,0,0],[0,1,0])
    zones={l:np.array([[.5,-.5],[.5+l,-.5],[.5+l,.5],[.5,.5]]) for l in pr['lengths_m']}
    rows=[]; gt_counts={c:0 for c in cats.values()}; max_orth=0.;max_roundtrip=0.;details=[]
    for idx,d in enumerate(data):
        token=d['token'];ego=np.array(d['ego2global']);lidar=np.array(d['lidar_points']['lidar2ego']);inv=np.linalg.inv(ego)
        for t in [ego,lidar]:
            assert np.allclose(t[3],[0,0,0,1]);err=np.max(np.abs(t[:3,:3].T@t[:3,:3]-np.eye(3)));max_orth=max(max_orth,float(err));assert err<1e-5
        gp=[]
        for gi,g in enumerate(d['instances']):
            label=g['bbox_label_3d']
            if label not in cats or g['num_lidar_pts']+g['num_radar_pts']<=0:continue
            c=cats[label]; b=np.array(g['bbox_3d'],float);center=(lidar@np.r_[b[:3],1])[:3]
            if np.linalg.norm(center[:2])>=ranges[c]:continue
            yaw=b[6];r=rotation([np.cos(yaw/2),0,0,np.sin(yaw/2)])
            p=hull(b[:3],b[3:6],r,lidar);gp.append((gi,c,p));gt_counts[c]+=1
        pp={}
        for m in pred:
            pp[m]=[]
            for pi,p in enumerate(pred[m][token]):
                assert p['sample_token']==token
                c=p['detection_name'];assert c in ranges
                center=inv@np.r_[p['translation'],1]
                max_roundtrip=max(max_roundtrip,float(np.max(np.abs(ego@center-np.r_[p['translation'],1]))))
                if np.linalg.norm(center[:2])>=ranges[c] or p['detection_score']<min(pr['thresholds']):continue
                w,l,h=p['size'];poly=hull(np.array(p['translation']),[l,w,h],rotation(p['rotation']),inv)
                pp[m].append((pi,float(p['detection_score']),poly))
        for length,zone in zones.items():
            gids=[gi for gi,c,p in gp if intersects(p,zone)];gt=bool(gids)
            for m in pred:
                hits=[(pi,s) for pi,s,p in pp[m] if intersects(p,zone)]
                for threshold in pr['thresholds']:
                    pids=[pi for pi,s in hits if s>=threshold];alert=bool(pids)
                    rows.append(dict(sample_token=token,scene_token=lookup[token][0],scene_name=lookup[token][1],length_m=length,threshold=threshold,model=m,gt_boxes=len(gids),pred_boxes=len(pids),gt_alert=int(gt),pred_alert=int(alert),miss=int(gt and not alert),unreferenced=int(not gt and alert)))
                    if length==5 and threshold==.3 and (gt or alert):details.append(dict(sample_token=token,model=m,gt_indices=gids,prediction_indices=pids))
        if (idx+1)%100==0:print('Frames analysed:',idx+1,flush=True)
    assert max_roundtrip<1e-8
    writer(out/'frame_results.csv',rows)
    summaries=[]; cis=[]; sceneids=sorted({v[0] for v in lookup.values()}); rng=np.random.default_rng(20261023); draws=rng.integers(0,150,(2000,150))
    for length,threshold in itertools.product(pr['lengths_m'],pr['thresholds']):
        arrays={}
        for m in pred:
            rr=[r for r in rows if r['length_m']==length and r['threshold']==threshold and r['model']==m]
            positive=sum(r['gt_alert'] for r in rr);negative=600-positive;miss=sum(r['miss'] for r in rr);false=sum(r['unreferenced'] for r in rr)
            summaries.append(dict(length_m=length,threshold=threshold,model=m,frames=600,positive_frames=positive,negative_frames=negative,correct_alerts=positive-miss,missed_alerts=miss,unreferenced_alerts=false,negative_no_alert=negative-false,miss_rate=ratio(miss,positive),unreferenced_rate=ratio(false,negative)))
            arr=np.array([[sum(r[k] for r in rr if r['scene_token']==s) for k in ['gt_alert','miss','unreferenced']] for s in sceneids]);arrays[m]=arr
        boots={m:arr[draws].sum(1) for m,arr in arrays.items()}
        for metric,num,den in [('miss_rate',1,0),('unreferenced_rate',2,None)]:
            vals={}
            for m,b in boots.items():
                ds=b[:,0] if den==0 else 600-b[:,0];v=np.divide(b[:,num],ds,out=np.full(2000,np.nan),where=ds>0);vals[m]=v
                valid=v[np.isfinite(v)];ci=np.percentile(valid,[2.5,97.5]) if len(valid) else [None,None]
                cis.append(dict(length_m=length,threshold=threshold,metric=metric,comparison=m,low=ci[0],high=ci[1],valid_resamples=len(valid)))
            diff=vals['fusion']-vals['lidar'];diff=diff[np.isfinite(diff)];ci=np.percentile(diff,[2.5,97.5]) if len(diff) else [None,None]
            cis.append(dict(length_m=length,threshold=threshold,metric=metric,comparison='fusion_minus_lidar',low=ci[0],high=ci[1],valid_resamples=len(diff)))
    writer(out/'summary.csv',summaries);writer(out/'scene_bootstrap.csv',cis)
    primary=[s for s in summaries if s['length_m']==5 and s['threshold']==.3]
    (out/'primary_results.json').write_text(json.dumps(primary,indent=2)+'\n');(out/'primary_alert_indices.json').write_text(json.dumps(details,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    base=[s for s in summaries if s['threshold']==.3 and s['model']=='lidar'];axs[0].bar([str(s['length_m']) for s in base],[s['positive_frames'] for s in base],color='#777777');axs[0].set(title='Annotation-positive frames',xlabel='Forward length (m)',ylabel='Frames / 600')
    for ax,metric,title in [(axs[1],'miss_rate','Missed alerts / positive frames'),(axs[2],'unreferenced_rate','Unreferenced alerts / negative frames')]:
        for m,color in [('lidar','#0072B2'),('fusion','#D55E00')]:
            rr=[s for s in summaries if s['threshold']==.3 and s['model']==m];ax.plot([s['length_m'] for s in rr],[100*s[metric] if s[metric] is not None else np.nan for s in rr],marker='o',label=m,color=color)
        ax.set(title=title,xlabel='Forward length (m)',ylabel='Percent');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('600-frame offline corridor study | score ≥ 0.30 | fixed 1 m-wide corridor',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    audit=dict(status='ANNOTATION_BASED_600_FRAME_CORRIDOR_PASS',samples=600,scenes=150,rows=len(rows),hashes=manifest,input_zip_sha256=sha(raw),protocol_sha256=sha((root/'protocol.json').read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),scene_selection_sha256=sha((root/'scene_selection.json').read_bytes()),gt_counts=gt_counts,max_rotation_orthogonality_error=max_orth,max_global_roundtrip_error_m=max_roundtrip,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,scope=pr['scope'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(primary,indent=2),flush=True)
if __name__=='__main__':main()
