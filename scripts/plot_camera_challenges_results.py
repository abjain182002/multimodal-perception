"""Recheck preserved evidence and reproduce compact camera challenge figures."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(root):
 e=root/'original_evidence'; manifest=json.loads((e/'evidence_manifest.json').read_text())
 for n,r in manifest['members'].items():
  p=e/n; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],n
 s=json.loads((e/'evaluation/seven_condition_evaluation.json').read_text())
 assert s['status']=='SEVEN_CONDITION_SUBSET_EVALUATION_AND_DRIVE_COPY_PASS'
 tokens=(e/'study/sample_tokens.txt').read_text().splitlines();assert len(set(tokens))==len(tokens)==600
 protocol=json.loads((e/'study/protocol.json').read_text());assert sha(e/'study/protocol.json')==s['protocol_sha256']
 names=[c['name'] for c in protocol['conditions']]; rec={r['condition']:r for r in s['conditions']}; assert set(names)==set(rec)
 clean=json.loads((e/'study/inference/clean/input_checks.json').read_text())
 metrics={}; deltas=[]; classrows=[]; errors=[]
 for name in names:
  r=rec[name]; p=e/'evaluation'/name/'metrics_summary.json';m=json.loads(p.read_text()); metrics[name]=m
  assert sha(p)==r['metrics_summary_sha256']; assert sha(e/'evaluation'/name/'metrics_details.json')==r['metrics_details_sha256']
  assert abs(np.mean(list(m['mean_dist_aps'].values()))-m['mean_ap'])<1e-12
  assert abs((5*m['mean_ap']+sum(max(0,1-v) for v in m['tp_errors'].values()))/10-m['nd_score'])<1e-12
  assert r['samples']==600 and r['scenes']==150 and r['filtered_gt_objects']==12318 and r['exact_prediction_gt_token_match']
  ledger=json.loads((e/'study/inference'/name/'input_checks.json').read_text());assert [x['sample_token'] for x in ledger]==tokens
  for a,b in zip(clean,ledger):
   for key in ('original_image_sha256','original_points_sha256','original_metadata_sha256','processed_points_sha256'): assert a[key]==b[key],(name,key)
   if name.startswith('calibration') or name=='clean':
    assert a['processed_image_sha256']==b['processed_image_sha256']
    assert a['transformed_image_sha256']==b['transformed_image_sha256']
   else: assert a['transformed_metadata_sha256']==b['transformed_metadata_sha256']
  base=rec['clean']; dm=100*(m['mean_ap']-base['mean_ap']);dn=100*(m['nd_score']-base['nd_score'])
  assert abs(dm-r['delta_map_points_vs_clean'])<1e-12 and abs(dn-r['delta_nds_points_vs_clean'])<1e-12
  deltas.append([name,600,100*m['mean_ap'],100*m['nd_score'],dm,dn])
  for cls,ap in m['mean_dist_aps'].items(): classrows.append([name,cls,100*ap,100*(ap-rec['clean']['class_ap'][cls])])
  for key,value in m['tp_errors'].items():errors.append([name,key,value])
 tables=root/'tables';figures=root/'figures';tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
 for filename,header,rows in [('headline_metrics.csv',['condition','samples','mAP_x100','NDS_x100','delta_mAP_points','delta_NDS_points'],deltas),('class_ap.csv',['condition','class','AP_x100','delta_AP_points'],classrows),('tp_errors.csv',['condition','metric','value'],errors)]:
  with (tables/filename).open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 families=[('Gaussian blur',[0,1,2],['clean','blur_sigma1','blur_sigma2'],'Gaussian sigma (pixels)'),('Additive Gaussian noise',[0,10,25],['clean','noise_std10','noise_std25'],'Noise standard deviation (pixel units)'),('Common extrinsic yaw error',[0,1,3],['clean','calibration_yaw1','calibration_yaw3'],'Positive yaw offset (degrees)')]
 fig,axes=plt.subplots(1,3,figsize=(11.5,3.65),sharey=True)
 for i,(title,x,conds,xlabel) in enumerate(families):
  ax=axes[i]
  for metric,label,color,marker in [('mean_ap','mAP','#0072B2','o'),('nd_score','NDS','#D55E00','s')]:
   y=[100*(metrics[n][metric]-metrics['clean'][metric]) for n in conds]
   ax.plot(x,y,marker=marker,color=color,label=label,lw=1.8)
   for xx,yy in zip(x[1:],y[1:]):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,-13 if metric=='mean_ap' else 7),textcoords='offset points',ha='center',fontsize=9)
  ax.axhline(0,color='.5',lw=.7);ax.set_xticks(x);ax.set_xlabel(xlabel,fontsize=9);ax.set_title(f'({chr(97+i)}) {title}',fontsize=10);ax.grid(axis='y',alpha=.22);ax.set_ylim(-5.9,.65)
 axes[0].set_ylabel('Change from fresh clean baseline (points)');axes[0].legend(loc='lower left',frameon=False)
 fig.suptitle('BEVFusion: synthetic camera challenges on the same 600 frames',fontsize=12)
 fig.tight_layout(rect=(0,.02,1,.94))
 for ext in ('png','pdf','svg'):fig.savefig(figures/f'comparison.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
 cls=list(metrics['clean']['mean_dist_aps']); degraded=names[1:]; matrix=np.array([[100*(metrics[n]['mean_dist_aps'][c]-metrics['clean']['mean_dist_aps'][c]) for c in cls] for n in degraded])
 fig,ax=plt.subplots(figsize=(10.7,4.4));lim=max(abs(matrix.min()),abs(matrix.max()));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
 ax.set_yticks(range(6),['Blur sigma 1','Blur sigma 2','Noise SD 10','Noise SD 25','Yaw +1 degree','Yaw +3 degrees']);ax.set_xticks(range(10),[c.replace('_',' ') for c in cls],rotation=35,ha='right')
 for i in range(6):
  for j in range(10):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>lim*.65 else 'black')
 ax.set_title('Supporting detail: class AP change from the study clean baseline');fig.colorbar(im,ax=ax,label='AP change (points)');fig.tight_layout()
 for ext in ('png','pdf','svg'):fig.savefig(figures/f'class_ap_delta.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
 (root/'analysis_summary.json').write_text(json.dumps({'status':'STORED_CAMERA_CHALLENGE_EVIDENCE_AND_ARITHMETIC_PASS','samples':600,'scenes':150,'conditions':7,'headline_metrics':deltas,'scope':'Manifest and stored arithmetic/input-ledger checks; no fresh GT replay or raw prediction rehash. All original 270 archive members preserved.'},indent=2)+'\n')
 print('Verified original manifest, seven metric summaries and 4200 input-ledger rows; generated two composite figures and three CSV tables.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/nuscenes/bevfusion/camera_challenges/20261016'));run(p.parse_args().root)
