"""CPU diagnostics of verified four-case export. Usage: python analyse.py --input input --output results
Input is the verified extracted export with nested input ZIP extracted into prepared/.
No model inference. Uses saved filtered GT; analyses only each case's focal class.
"""
import argparse,json,csv,hashlib,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();I=a.input;O=a.output;O.mkdir(parents=True,exist_ok=True)
ref=I/'prepared/reference/nuscenes_six_case_annotation_inputs.zip'
assert hashlib.sha256(ref.read_bytes()).hexdigest()=='574a5ea93c4fc455317562caaa82a30f25ce3f8b20715ddbf2e05c0cff0b9bdd'
with zipfile.ZipFile(ref) as z:
 def rd(n):
  names=[s for s in z.namelist() if s==n or s.endswith('/'+n)];assert len(names)==1,(n,names);return json.loads(z.read(names[0]))
 C=rd('candidate_contexts.json');SD={x['filename']:x for x in rd('sample_data.json')};EP={x['token']:x for x in rd('ego_pose.json')};CS={x['token']:x for x in rd('calibrated_sensor.json')}
cases=[('A','scene-0332','b92ee1ed90c242d18bce9a8fdbbeb8d8',22,'CAM_BACK_LEFT'),('B','scene-0562','0f483d8f22694132946c831f5d644cd7',26,'CAM_FRONT_LEFT'),('C','scene-0913','3b156a6793924c4497f394a3e34ddbf0',23,'CAM_FRONT_RIGHT'),('D','scene-0092','e13ec49d20974a01942b83317fcbfa46',7,'CAM_FRONT_LEFT')]
conditions=[('lidar','clean',1),('lidar_camera','clean',1),('lidar_camera','darkening_050',.5),('lidar_camera','darkening_025',.25)]
P={(m,k):json.loads((I/'inference'/m/k/'results_nusc.json').read_text())['results'] for m,k,g in conditions}
def dist(p,g):return float(np.linalg.norm(np.array(p['translation'][:2])-g['translation'][:2]))
def matches(gt,pred,th):
 used=set();pairs={}
 for j in sorted(range(len(pred)),key=lambda j:(-pred[j]['score'],j)):
  if pred[j]['score']<th:continue
  dd,gi=min(((dist(pred[j],g),i) for i,g in gt.items() if i not in used),default=(float('inf'),-1))
  if dd<2:used.add(gi);pairs[gi]=j
 return pairs

def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def project(b,sd):
 w,l,h=b['size'];v=np.array([[x*l/2,y*w/2,z*h/2] for z in [-1,1] for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]]).T
 v=R(b['rotation'])@v+np.array(b['translation'])[:,None]
 for pose in [EP[sd['ego_pose_token']],CS[sd['calibrated_sensor_token']]]:v=R(pose['rotation']).T@(v-np.array(pose['translation'])[:,None])
 assert v[2].min()>0
 uv=np.array(CS[sd['calibrated_sensor_token']]['camera_intrinsic'])@v
 return uv[:2]/uv[2]
edges=[(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]
rows=[];comparisons=[];panels=[]
for letter,scene,t,gi,cam in cases:
 c=C[t];g=c['ground_truth'][gi];cls=g['class'];gt={i:x for i,x in enumerate(c['ground_truth']) if x['class']==cls}
 ego=EP[SD[c['sensor_files']['LIDAR_TOP']]['ego_pose_token']]['translation'];maxrange=30 if cls=='traffic_cone' else 50
 filename=c['sensor_files'][cam];sd=SD[filename];im=np.array(Image.open(I/'prepared/dataset'/filename).convert('RGB'));uv=project(g,sd);lo,hi=uv.min(1),uv.max(1);center=(lo+hi)/2;span=max(float(max(hi-lo))*1.7,180);width=min(span*1.65,im.shape[1]);height=min(span,im.shape[0]);cx=np.clip(center[0],width/2,im.shape[1]-width/2);cy=np.clip(center[1],height/2,im.shape[0]-height/2)
 casepanels=[]
 for m,k,gain in conditions:
  raw=P[m,k][t];pred=[dict(x,score=x['detection_score'],**{'class':x['detection_name']}) for x in raw if x['detection_name']==cls and np.linalg.norm(np.array(x['translation'][:2])-ego[:2])<maxrange]
  dd,pi=min((dist(x,g),j) for j,x in enumerate(pred));p0=pred[pi];states={th:matches(gt,pred,th) for th in [.1,.3,.5]}
  for th,s in states.items():assert s.get(gi) in [None,pi],(letter,m,k,th,'assignment competition')
  row=dict(case=letter,scene=scene,sample_token=t,gt_index=gi,class_name=cls,camera=cam,model=m,condition=k,image_gain=gain,nearest_score=p0['score'],nearest_xy_error_m=dd,matched_010=gi in states[.1],matched_030=gi in states[.3],matched_050=gi in states[.5]);rows.append(row)
  if k=='clean':
   old=[x for x in c['predictions'][m] if x['class']==cls];od,oi=min((dist(x,g),j) for j,x in enumerate(old));os=matches(gt,old,.3)
   assert (gi in os)==row['matched_030']
   comparisons.append(dict(case=letter,model=m,original_score=old[oi]['score'],new_score=p0['score'],score_delta=p0['score']-old[oi]['score'],original_error_m=od,new_error_m=dd,error_delta_m=dd-od,outcome_reproduced=True))
  casepanels.append((row,im,uv,project(p0,sd),(cx-width/2,cx+width/2,cy+height/2,cy-height/2)))
 panels.append(casepanels)

def draw(ax,panel):
 r,im,uv,pv,lim=panel;ax.imshow(np.clip(im.astype(float)*r['image_gain'],0,255).astype('uint8'))
 for xy,col,style in [(uv,'#ff35a0','-'),(pv,'#00dcff','-' if r['matched_030'] else '--')]:
  for j,k in edges:ax.plot(xy[0,[j,k]],xy[1,[j,k]],color=col,ls=style,lw=1.15)
 ax.set_xlim(lim[:2]);ax.set_ylim(lim[2:]);ax.set_xticks([]);ax.set_yticks([])
 ax.set_xlabel(f"score {r['nearest_score']:.3f} | XY error {r['nearest_xy_error_m']:.3f} m\n{'MATCH' if r['matched_030'] else 'MISS'} at score 0.30",fontsize=9)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axs=plt.subplots(4,3,figsize=(12,12));fig.subplots_adjust(top=.91,bottom=.10,hspace=.62,wspace=.10)
for i,ps in enumerate(panels):
 for j,pa in enumerate(ps[1:]):
  draw(axs[i,j],pa);r=pa[0];axs[i,j].set_title(f"{r['case']}: {r['class_name'].replace('_',' ')} | gain {r['image_gain']:g}",fontsize=11)
fig.suptitle('Four fixed cases: fusion under synthetic image darkening',fontsize=17,y=.98)
fig.text(.5,.945,'Pink: ground truth | cyan: nearest same-class candidate (dashed below threshold)',ha='center')
fig.text(.5,.012,'Identical crop per case. Display gains applied to full-resolution images; model gains applied after resize/crop.\nSelected objects, not population accuracy. Projections omit motion compensation and occlusion tests.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(O/f'four_case_darkening_composite.{ext}',dpi=170)
plt.close(fig)
for ps in panels:
 fig,axs=plt.subplots(1,4,figsize=(16,4.3));fig.subplots_adjust(top=.79,bottom=.20,wspace=.10)
 for ax,pa in zip(axs,ps):
  draw(ax,pa);r=pa[0];ax.set_title(('LiDAR clean' if r['model']=='lidar' else f"Fusion gain {r['image_gain']:g}"))
 fig.suptitle(f"Case {ps[0][0]['case']} | {ps[0][0]['scene']} | {ps[0][0]['camera']}",fontsize=15)
 fig.text(.5,.025,'Pink GT; cyan nearest candidate; dashed = below 0.30. Display crops illustrate gain; not exact packed model tensors.',ha='center',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(O/f"case_{ps[0][0]['case']}.{ext}",dpi=170)
 plt.close(fig)
for name,data in [('object_evidence.csv',rows),('clean_reproduction.csv',comparisons)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(O/'audit.json').write_text(json.dumps(dict(status='FOUR_CASE_FOCAL_OBJECT_ANALYSIS_PASS',objects=4,condition_object_records=16,thresholds=[.1,.3,.5],matching='Score-ordered one-to-one same-class greedy match; XY distance <2 m; score >= threshold',clean_highlighted_outcomes_reproduced=8,scope='Focal objects only. No full-frame mAP/NDS; no bitwise clean prediction equivalence claimed.'),indent=2))
print(json.dumps(rows,indent=2))
