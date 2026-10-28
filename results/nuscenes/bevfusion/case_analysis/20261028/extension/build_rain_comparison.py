"""Two outcome-selected rain-labelled scenes; observational comparisons, no weather causality."""
import json,csv,argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def rd(n):return json.loads((a.reference/n).read_text())
C=rd('candidate_contexts.json');SD={x['filename']:x for x in rd('sample_data.json')};EP={x['token']:x for x in rd('ego_pose.json')};CS={x['token']:x for x in rd('calibrated_sensor.json')};SR=rd('sensor_records.json')
def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def project(b,sd):
 w,l,h=b['size'];v=np.array([[x*l/2,y*w/2,z*h/2] for z in [-1,1] for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]]).T;v=R(b['rotation'])@v+np.array(b['translation'])[:,None]
 for pose in [EP[sd['ego_pose_token']],CS[sd['calibrated_sensor_token']]]:v=R(pose['rotation']).T@(v-np.array(pose['translation'])[:,None])
 assert v[2].min()>0;v=np.array(CS[sd['calibrated_sensor_token']]['camera_intrinsic'])@v;return v[:2]/v[2]
def distance(x,g):return float(np.linalg.norm(np.array(x['translation'][:2])-g['translation'][:2]))
chosen=[('C','scene-0913','3b156a6793924c4497f394a3e34ddbf0',23,'CAM_FRONT_RIGHT'),('E','scene-0626','bbf1d23e80b84da798c56cbac090f7d9',0,'CAM_FRONT')];rows=[]
fig,axs=plt.subplots(2,2,figsize=(10,8));fig.subplots_adjust(top=.86,bottom=.10,hspace=.55,wspace=.08)
for row,(letter,scene,t,gi,cam) in enumerate(chosen):
 c=C[t];g=c['ground_truth'][gi];filename=c['sensor_files'][cam];sd=SD[filename];im=Image.open(a.reference/'dataset'/filename);import hashlib
 assert hashlib.sha256((a.reference/'dataset'/filename).read_bytes()).hexdigest()==SR[filename]['sha256']
 uv=project(g,sd);lo,hi=uv.min(1),uv.max(1);ctr=(lo+hi)/2;span=max(max(hi-lo)*1.7,180);w=min(span*1.65,im.width);h=min(span,im.height);cx=np.clip(ctr[0],w/2,im.width-w/2);cy=np.clip(ctr[1],h/2,im.height-h/2)
 for col,m in enumerate(['lidar','lidar_camera']):
  dd,pi=min((distance(x,g),i) for i,x in enumerate(c['predictions'][m]) if x['class']==g['class']);b=c['predictions'][m][pi];assigned=next((z for z in c['matching'][m]['matches'] if z['ground_truth_index']==gi),None);assert assigned is None or assigned['prediction_index']==pi
  ax=axs[row,col];ax.imshow(im)
  for vv,color,ls in [(uv,'#ff35a0','-'),(project(b,sd),'#00d9ff','-' if assigned else '--')]:
   for j,k in [(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]:ax.plot(vv[0,[j,k]],vv[1,[j,k]],color=color,ls=ls,lw=1.2)
  ax.set_xlim(cx-w/2,cx+w/2);ax.set_ylim(cy+h/2,cy-h/2);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"{letter}: {scene} | {'LiDAR' if col==0 else 'Fusion'}")
  ax.set_xlabel(f"{g['class']} | score {b['score']:.3f} | XY error {dd:.3f} m\n{'MATCH' if assigned else 'MISS'} at 0.30",fontsize=10)
  rows.append(dict(case=letter,scene=scene,sample_token=t,gt_index=gi,class_name=g['class'],camera=cam,model=m,score=b['score'],xy_error_m=dd,matched_030=bool(assigned)))
fig.suptitle('Rain-labelled scenes: a shared success and a fusion regression',fontsize=14,y=.97)
fig.text(.5,.91,'Identical camera crop for each model | pink GT; cyan candidate; dashed below threshold',ha='center',fontsize=10)
fig.text(.5,.025,'Scene descriptions include rain; frame rain intensity is not measured. Different objects/scenes prevent causal weather attribution.\nOriginal preserved clean predictions; case E adds no new inference. Projections omit motion compensation and occlusion tests.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(a.output/f'real_rain_case_comparison.{ext}',dpi=170)
with (a.output/'real_rain_object_evidence.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
