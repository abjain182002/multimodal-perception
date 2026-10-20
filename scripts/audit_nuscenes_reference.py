"""Bounded comparison of saved BEVFusion inference configs and primary references.

Uses a restricted AST reader for dictionary configuration files. Does not execute
configuration code, rerun inference, or identify a causal explanation of score gaps.
"""
import ast
import copy
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import urllib.request
import zipfile

def value(node, env):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.Name):return copy.deepcopy(env[node.id])
    if isinstance(node,ast.List):return [value(x,env) for x in node.elts]
    if isinstance(node,ast.Tuple):return tuple(value(x,env) for x in node.elts)
    if isinstance(node,ast.Dict):return {value(k,env):value(v,env) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand,env)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return value(node.left,env)+value(node.right,env)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):return value(node.left,env)*value(node.right,env)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):return value(node.left,env)/value(node.right,env)
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
        assert not node.args
        return {k.arg:value(k.value,env) for k in node.keywords}
    raise ValueError(ast.dump(node))

def merge(a,b):
    if not isinstance(a,dict) or not isinstance(b,dict):return copy.deepcopy(b)
    out=copy.deepcopy(a)
    for k,v in b.items():out[k]=merge(out.get(k),v)
    return out

def parse(path, inherited=None):
    env=copy.deepcopy(inherited or {})
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n,ast.Delete):continue
        assert isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        key=n.targets[0].id
        if key=='_base_':continue
        new=value(n.value,env)
        env[key]=merge(env.get(key),new)
    return env

def diff(a,b,path=''):
    rows=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:rows.append((path+'.'+k,a.get(k,'<ABSENT>'),b.get(k,'<ABSENT>')))
            else:rows+=diff(a[k],b[k],path+'.'+k)
    elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):rows+=diff(x,y,f'{path}[{i}]')
    elif a!=b:rows.append((path,a,b))
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-archive',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    s=args.source_dir;s.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    pin='fe25f7a51d36e3702f961e198894580d83c4387b'
    raw=f'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/{pin}/'
    download='https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/'
    urls={
      'upstream_lidar.py':raw+'projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
      'upstream_fusion.py':raw+'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
      'upstream_lidar.log':download+'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log',
      'upstream_fusion.log':download+'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log',
      'nuscenes_metric.py':raw+'mmdet3d/evaluation/metrics/nuscenes_metric.py',
      'upstream_readme.md':raw+'projects/BEVFusion/README.md',
    }
    sources={}
    for name,url in urls.items():
        path=s/name
        if not path.is_file():path.write_bytes(urllib.request.urlopen(url,timeout=30).read())
        data=path.read_bytes();sources[name]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    assert "eval_version: str = 'detection_cvpr_2019'" in (s/'nuscenes_metric.py').read_text()
    lidar=parse(s/'upstream_lidar.py'); fusion=parse(s/'upstream_fusion.py',lidar)
    rows=[];logged_rows=[];validation_rows=[];checks=[];environment={}
    for model in ['lidar','fusion']:
        lines=(s/f'upstream_{model}.log').read_text().splitlines()
        start=next(i for i,l in enumerate(lines) if 'INFO - Config:' in l)+1
        end=next(i for i in range(start,len(lines)) if re.match(r'^2023/\d\d/\d\d ',lines[i]))
        (s/f'logged_{model}_config.py').write_text('\n'.join(lines[start:end])+'\n')
        environment[model]={}
        for key in ['Python','PyTorch','MMEngine','CUDA_HOME']:
            line=next((l.strip() for l in lines[:40] if l.strip().startswith(key+':')),None)
            if line:environment[model][key]=line.split(':',1)[1].strip()
        environment[model]['gpu_line']=next(l.strip() for l in lines[:40] if 'NVIDIA A100' in l)
        environment[model]['upstream_nuscenes_devkit_version']='Not specified in inspected linked log'
        for line in lines:
            if 'Epoch(val)' in line and '/NDS:' in line:
                validation_rows.append(dict(model='lidar_camera' if model=='fusion' else model,
                    epoch=int(re.search(r'Epoch\(val\) \[(\d+)\]',line).group(1)),
                    logged_mAP=float(re.search(r'/mAP: ([0-9.]+)',line).group(1)),
                    logged_NDS=float(re.search(r'/NDS: ([0-9.]+)',line).group(1))))
    expected_sha='37a5882333283e882d21d8f165b308f79a5039a8e1064334ad480b93bd665269'
    assert hashlib.sha256(args.original_archive.read_bytes()).hexdigest()==expected_sha
    checkpoint=json.loads(args.checkpoint_audit.read_text())
    with zipfile.ZipFile(args.original_archive) as archive:
        manifest=json.loads(archive.read('evidence_manifest.json'))['members']
        def read_original(name):
            data=archive.read(name);r=manifest[name]
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
            sources['project:'+name]={'bytes':len(data),'sha256':r['sha256']}
            return data
        env=json.loads(read_original('evaluation/environment_audit.json'))
        cfg=json.loads(read_original('evaluation/detection_config.json'))
        for model,up in [('lidar',lidar),('lidar_camera',fusion)]:
            for batch in range(1,4):
                pre=f'inference/batch_{batch:02d}/'
                protocol=json.loads(read_original(pre+model+'/protocol.json'))
                config_bytes=read_original(pre+model+'/resolved_config.py')
                assert hashlib.sha256(config_bytes).hexdigest()==protocol['resolved_config_sha256']
                assert protocol['checkpoint_sha256']==checkpoint[model]['sha256']
                assert protocol['source_commit']==pin
                path=s/f'project_{batch:02d}_{model}.py';path.write_bytes(config_bytes);actual=parse(path)
                assert actual['model']==up['model']
                assert actual['test_pipeline']==up['test_pipeline']
                loader_diff=diff(up['test_dataloader'],actual['test_dataloader'],'test_dataloader')
                assert {p for p,a,b in loader_diff}=={
                    'test_dataloader.dataset.ann_file','test_dataloader.dataset.data_root','test_dataloader.dataset.metainfo.version'}
                for p,a,b in loader_diff:rows.append(dict(model=model,batch=batch,path=p,upstream=a,project=b))
                sweep=next(t for t in actual['test_pipeline'] if t['type']=='LoadPointsFromMultiSweeps')
                assert sweep['sweeps_num']==9
                checks.append(dict(model=model,batch=batch,model_config_equal=True,test_pipeline_equal=True,
                    checkpoint_hash_matches_download_record=True,preceding_sweeps=9))
                if batch==1:
                    log_model='fusion' if model=='lidar_camera' else model
                    logged=parse(s/f'logged_{log_model}_config.py')
                    for key in ['model','test_pipeline']:
                        logged_rows.extend(dict(model=model,path=p,logged_reference=a,project=b)
                                           for p,a,b in diff(logged[key],actual[key],key))
            assert all('strict=True' in read_original(f'inference/batch_{b:02d}/inference.py').decode() for b in range(1,4))
            summary=json.loads(read_original(f'evaluation/{model}/metrics_summary.json'))
            best=max((r for r in validation_rows if r['model']==model),key=lambda r:r['logged_mAP'])
            best['project_mAP']=summary['mean_ap'];best['project_NDS']=summary['nd_score']
            best['project_minus_logged_mAP_points']=100*(summary['mean_ap']-best['logged_mAP'])
            best['project_minus_logged_NDS_points']=100*(summary['nd_score']-best['logged_NDS'])
    outputs={'audit.json':dict(status='REFERENCE_AUDIT_COMPLETE_CAUSE_UNRESOLVED',checks=checks,
                project_evaluator_environment=env,upstream_logged_environment=environment,
                checkpoint_hashes=checkpoint,detection_config=cfg,
                cause='No cause established; recorded core configs and pipelines match pinned reference',
                scope='Static reference/provenance audit; no inference, upstream-environment rerun or upstream prediction replay.'),
             'config_differences.json':rows,'logged_config_differences.json':logged_rows,
             'source_manifest.json':sources}
    for name,data in outputs.items():(args.output/name).write_text(json.dumps(data,indent=2)+'\n')
    with (args.output/'upstream_validation_rows.csv').open('w',newline='') as f:
        keys=sorted(set().union(*(r.keys() for r in validation_rows)))
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(validation_rows)
    print('REFERENCE AUDIT COMPLETE: six model/batch model configurations and test pipelines match; cause unresolved.')
