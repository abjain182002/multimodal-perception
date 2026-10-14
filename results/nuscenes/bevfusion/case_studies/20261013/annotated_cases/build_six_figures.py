from pathlib import Path
import json,hashlib,zipfile,collections,textwrap
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).parent; D=P/'six_inputs'; O=P/'annotated_cases';O.mkdir(exist_ok=True)
def read(n):return json.loads((D/n).read_text())
C=read('candidate_contexts.json'); SD={r['filename']:r for r in read('sample_data.json')}; EP={r['token']:r for r in read('ego_pose.json')}; CS={r['token']:r for r in read('calibrated_sensor.json')}; SEL=read('selection.json')['cases'];ROWS={r['sample_token']:r for a in read('case_candidates.json').values() for r in a}
def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def corners(b):
 w,l,h=b['size'];a=np.array([[x*l/2,y*w/2,z*h/2] for z in [-1,1] for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]]).T
 return R(b['rotation'])@a+np.array(b['translation'])[:,None]
def transform(p,pose):return R(pose['rotation']).T@(p-np.array(pose['translation'])[:,None])
def camera(b,sd):
 cs=CS[sd['calibrated_sensor_token']];pts=transform(transform(corners(b),EP[sd['ego_pose_token']]),cs)
 if np.min(pts[2])<=.2:return None
 uv=np.array(cs['camera_intrinsic'])@pts;uv=uv[:2]/uv[2];w,h=sd['width'],sd['height'];lo=np.maximum(uv.min(1),[0,0]);hi=np.minimum(uv.max(1),[w,h]);area=np.prod(np.maximum(hi-lo,0))
 if area<80 or np.any(uv.min(1)<[5,5]) or np.any(uv.max(1)>[w-5,h-5]):return None
 return area,uv,lo,hi
edges=[(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]
records=[];colors=['#CC0077','#E69F00','#0072B2']; plt.rcParams.update({'font.size':10,'axes.titlesize':11})
for num,(t,category) in enumerate(SEL.items(),1):
 c=C[t]; row=ROWS[t];gt=c['ground_truth']; lm={v['ground_truth_index']:v for v in c['matching']['lidar']['matches']};fm={v['ground_truth_index']:v for v in c['matching']['lidar_camera']['matches']};l=set(lm);f=set(fm)
 targets={'fusion_helps':f-l,'fusion_regresses':l-f,'shared_success':l&f,'shared_miss':set(range(len(gt)))-l-f}[category]
 options=[]
 for gi in sorted(targets):
  views=[]
  for channel,filename in c['sensor_files'].items():
   if channel.startswith('CAM_'):
    res=camera(gt[gi],SD[filename])
    if res:views.append((res[0],channel,filename,res))
  if views:
   best=max(views,key=lambda x:(x[0],x[1]));options.append((gi,best))
 options.sort(key=lambda x:(-x[1][0],x[0]));chosen=[];classes=set()
 for op in options:
  cl=gt[op[0]]['class']
  if cl not in classes:chosen.append(op);classes.add(cl)
  if len(chosen)==3:break
 for op in options:
  if len(chosen)==3:break
  if all(op[0]!=x[0] for x in chosen):chosen.append(op)
 assert chosen
 fig=plt.figure(figsize=(16,11),facecolor='white');grid=fig.add_gridspec(2,6,height_ratios=[1,1.7],left=.055,right=.975,bottom=.19,top=.85,hspace=.3,wspace=.35)
 pose=EP[SD[c['sensor_files']['LIDAR_TOP']]['ego_pose_token']]
 evidence=[]
 for j,(gi,best) in enumerate(chosen):
  _,channel,filename,res=best;_,uv,lo,hi=res;ax=fig.add_subplot(grid[0,j*2:j*2+2]);im=Image.open(D/'dataset'/filename);ax.imshow(im)
  for a,b in edges:ax.plot(uv[0,[a,b]],uv[1,[a,b]],c=colors[j],lw=1.6)
  centre=(lo+hi)/2;span=max(float(np.max(hi-lo))*1.7,180);width=min(span*1.5,im.width);height=min(span,im.height);cx=np.clip(centre[0],width/2,im.width-width/2);cy=np.clip(centre[1],height/2,im.height-height/2)
  ax.set_xlim(cx-width/2,cx+width/2);ax.set_ylim(cy+height/2,cy-height/2);ax.set_title(f'{j+1}. {gt[gi]["class"].replace("_"," ")} | {channel[4:]}',color=colors[j]);ax.set_xticks([]);ax.set_yticks([])
  ax.set_xlabel('LiDAR: '+('matched' if gi in l else 'missed')+'  |  Fusion: '+('matched' if gi in f else 'missed'))
  evidence.append({'label':j+1,'gt_index':gi,'class':gt[gi]['class'],'camera':channel,'lidar_match':lm.get(gi),'fusion_match':fm.get(gi)})
  im.close()
 for k,model in enumerate(['lidar','lidar_camera']):
  ax=fig.add_subplot(grid[1,k*3:k*3+3]);matches=lm if k==0 else fm
  for b in gt:
   p=transform(corners(b),pose);idx=[0,1,2,3,0];ax.plot(-p[1,idx],p[0,idx],c='#C8CDD2',lw=.6)
  for j,(gi,_) in enumerate(chosen):
   p=transform(corners(gt[gi]),pose);idx=[0,1,2,3,0];ax.plot(-p[1,idx],p[0,idx],c=colors[j],lw=2,ls='--');cen=transform(np.array(gt[gi]['translation'])[:,None],pose).ravel();ax.annotate(str(j+1),(-cen[1],cen[0]),xytext=(5,6),textcoords='offset points',color=colors[j],weight='bold',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
   if gi in matches:
    pred=c['predictions'][model][matches[gi]['prediction_index']];p=transform(corners(pred),pose);ax.plot(-p[1,idx],p[0,idx],c=colors[j],lw=1.5)
   else:ax.scatter([-cen[1]],[cen[0]],marker='x',s=60,c=colors[j],linewidths=2)
  ax.scatter([0],[0],marker='^',c='#202E40',s=50);ax.set_xlim(-55,55);ax.set_ylim(-55,55);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('Right of ego vehicle (m)');ax.set_ylabel('Forward of ego vehicle (m)');m=c['matching'][model];ax.set_title(('LiDAR-only' if k==0 else 'Camera–LiDAR')+f"  |  All-class TP {len(m['matches'])}, FP {len(m['false_positive_indices'])}, FN {len(m['false_negative_indices'])}")
  ax.legend(handles=[Line2D([0],[0],c='black',ls='--',label='Highlighted ground truth'),Line2D([0],[0],c='black',label='Matched prediction'),Line2D([0],[0],c='black',marker='x',ls='',label='Missed at score ≥0.30')],loc='lower left',fontsize=8)
 counts=dict(collections.Counter(gt[i]['class'] for i in targets));details=', '.join(f'{v} {k.replace("_"," ")}' for k,v in counts.items());title={'fusion_helps':'Fusion-only matches','fusion_regresses':'LiDAR-only matches','shared_success':'Shared matches','shared_miss':'Shared misses'}[category]
 fig.suptitle(f'CASE {num:02d}  |  {row["scene_name"]}  |  {title}',x=.055,y=.97,ha='left',fontsize=20,weight='bold',color='#162B45');fig.text(.055,.92,textwrap.fill(f'Target outcome in this frame: {len(targets)} objects ({details}).',125),fontsize=11)
 fig.text(.055,.09,'Numbered crops show projected ground-truth boxes. BEV displays selected matches; other GT boxes are grey.\nScore ≥0.30; same-class centre distance <2 m. Crop selection: projected box area, prioritising class diversity.',fontsize=9)
 fig.text(.055,.045,'Illustrative, outcome-selected case; not a causal test. Projection uses camera-time ego pose with sample-time boxes;\nno motion compensation or occlusion test. A projected box does not establish that an object is visually unobstructed.',fontsize=9,color='#555555')
 name=f'case_{num:02d}_{row["scene_name"]}';fig.savefig(O/(name+'.png'),dpi=170);plt.close(fig)
 records.append({'case':num,'scene':row['scene_name'],'token':t,'category':category,'target_count':len(targets),'target_classes':counts,'highlighted_objects':evidence,'lidar_tp':len(l),'fusion_tp':len(f),'figure':name+'.png'})
 print(name,[(e['class'],e['camera']) for e in evidence],flush=True)
(O/'annotation_records.json').write_text(json.dumps(records,indent=2)+'\n')
