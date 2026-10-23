"""Offline hypothetical mobile-robot corridor alerts. No inference or robot control."""
import argparse, csv, hashlib, itertools, json, platform, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull

def rotation(q):
    q=np.asarray(q,float); assert abs(np.linalg.norm(q)-1)<1e-5
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def footprint(b,pose):
    w,l,h=b['size']; assert min(w,l,h)>0
    corners=np.array(list(itertools.product([-l/2,l/2],[-w/2,w/2],[-h/2,h/2])))
    glob=corners@rotation(b['rotation']).T+np.array(b['translation'])
    r=rotation(pose['rotation']); t=np.array(pose['translation'])
    ego=(glob-t)@r
    assert np.max(np.abs(ego@r.T+t-glob))<1e-9
    xy=ego[:,:2]; return xy[ConvexHull(xy).vertices]

def intersects(a,b):
    # Separating-axis test on two convex polygons; touching counts as overlap.
    for p in (a,b):
        for edge in np.roll(p,-1,axis=0)-p:
            axis=np.array([-edge[1],edge[0]]); aa=a@axis; bb=b@axis
            if aa.max()<bb.min()-1e-10 or bb.max()<aa.min()-1e-10: return False
    return True

def match(gt,preds,threshold):
    available=set(range(len(gt))); pairs={}
    for pi in sorted((i for i,p in enumerate(preds) if p['score']>=threshold),key=lambda i:(-preds[i]['score'],i)):
        p=preds[pi]
        choices=[(np.linalg.norm(np.array(p['translation'][:2])-g['translation'][:2]),gi) for gi,g in enumerate(gt) if gi in available and g['class']==p['class']]
        if choices:
            d,gi=min(choices)
            if d<2: pairs[gi]=pi; available.remove(gi)
    return pairs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',required=True); ap.add_argument('--output',default='results'); args=ap.parse_args()
    root=Path(__file__).resolve().parent; protocol=json.loads((root/'protocol.json').read_text())
    raw=Path(args.inputs).read_bytes(); assert hashlib.sha256(raw).hexdigest()==protocol['input_archive_sha256']
    z=zipfile.ZipFile(args.inputs); read=lambda n:json.loads(z.read(n))
    contexts=read('candidate_contexts.json'); selection=read('selection.json')['cases']
    sd=read('sample_data.json'); poses={p['token']:p for p in read('ego_pose.json')}
    candidates={r['sample_token']:r for group in read('case_candidates.json').values() for r in group}
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    # Geometry fixtures: disjoint, touching, containment, known 90-degree rotation.
    square=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    assert intersects(square,square+np.array([1,0]))
    assert not intersects(square,square+np.array([1.01,0]))
    assert intersects(square,square*.1+.2)
    assert np.allclose(rotation([2**-.5,0,0,2**-.5])@np.array([1,0,0]),[0,1,0])
    rows=[]; geometry={}; pose_records=[]
    fig,axes=plt.subplots(2,3,figsize=(11,8),layout='constrained')
    half=protocol['robot_width_m']/2+protocol['lateral_margin_each_side_m']; start=protocol['corridor_start_x_m']
    for ax,token in zip(axes.flat,selection):
        c=contexts[token]; records=[s for s in sd if s['sample_token']==token and s['filename']==c['sensor_files']['LIDAR_TOP']]
        assert len(records)==1 and records[0]['is_key_frame']; pose=poses[records[0]['ego_pose_token']]
        pose_records.append(dict(sample_token=token,sample_data_token=records[0]['token'],ego_pose=pose))
        boxes={'ground_truth':c['ground_truth'],**c['predictions']}
        polys={m:[footprint(b,pose) for b in bb] for m,bb in boxes.items()}
        geometry[token]={m:[p.tolist() for p in pp] for m,pp in polys.items()}
        for length,threshold in itertools.product(protocol['forward_lengths_m'],protocol['score_thresholds']):
            zone=np.array([[start,-half],[start+length,-half],[start+length,half],[start,half]])
            gt_ids={i for i,p in enumerate(polys['ground_truth']) if intersects(p,zone)}
            for model in ['lidar','lidar_camera']:
                pred_ids={i for i,p in enumerate(polys[model]) if boxes[model][i]['score']>=threshold and intersects(p,zone)}
                pairs=match(boxes['ground_truth'],boxes[model],threshold)
                if threshold==.3: assert pairs=={p['ground_truth_index']:p['prediction_index'] for p in c['matching'][model]['matches']}
                covered={gi for gi in gt_ids if pairs.get(gi) in pred_ids}
                outcome=('correct_alert' if pred_ids else 'missed_alert') if gt_ids else ('unreferenced_alert' if pred_ids else 'no_annotated_obstacle_no_alert')
                rows.append(dict(scene=candidates[token]['scene_name'],sample_token=token,length_m=length,score_threshold=threshold,model=model,zone_gt=len(gt_ids),zone_predictions=len(pred_ids),covered_zone_gt=len(covered),missed_zone_gt=len(gt_ids-covered),alert_outcome=outcome,gt_indices=json.dumps(sorted(gt_ids)),prediction_indices=json.dumps(sorted(pred_ids))))
        for model,color,style in [('ground_truth','black','-'),('lidar','#0072B2','--'),('lidar_camera','#D55E00',':')]:
            for b,p in zip(boxes[model],polys[model]):
                if model!='ground_truth' and b['score']<.3: continue
                ax.add_patch(Polygon(p[:,[1,0]],closed=True,fill=False,edgecolor=color,linestyle=style,lw=1.2,alpha=.8))
        ax.add_patch(Rectangle((-half,start),2*half,5,facecolor='#009E73',alpha=.15,edgecolor='#009E73'))
        ax.add_patch(Rectangle((-.4,-.5),.8,1,facecolor='gray'))
        primary=[r for r in rows if r['sample_token']==token and r['length_m']==5 and r['score_threshold']==.3]
        ax.set_title(candidates[token]['scene_name']+' | zone GT '+str(primary[0]['zone_gt'])+'\nAlerts L/F: '+('/'.join('yes' if r['zone_predictions'] else 'no' for r in primary)),fontsize=10)
        ax.set(xlim=(6,-6),ylim=(-1,12),xlabel='Left y (m)',ylabel='Forward x (m)'); ax.set_aspect('equal'); ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=c,ls=s,label=l) for c,s,l in [('black','-','Ground truth'),('#0072B2','--','LiDAR'),('#D55E00',':','Camera + LiDAR'),('#009E73','-','5 m alert corridor')]],loc='outside upper center',ncol=4)
    fig.savefig(out/'corridor_cases.pdf'); fig.savefig(out/'corridor_cases.png',dpi=180); plt.close(fig)
    with (out/'all_conditions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'ego_footprints.json').write_text(json.dumps(geometry,separators=(',',':'))+'\n')
    (out/'pose_records.json').write_text(json.dumps(pose_records,indent=2)+'\n')
    primary=[r for r in rows if r['length_m']==5 and r['score_threshold']==.3]
    (out/'primary_results.json').write_text(json.dumps(primary,indent=2)+'\n')
    audit=dict(status='SIX_CASE_OFFLINE_CORRIDOR_REPLAY_PASS',samples=len(selection),rows=len(rows),geometry_fixtures_pass=True,all_box_global_ego_roundtrips_below_1e_9_m=True,saved_matching_replayed=True,protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(raw).hexdigest(),python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,scope=protocol['limits'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(primary,indent=2)); print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
