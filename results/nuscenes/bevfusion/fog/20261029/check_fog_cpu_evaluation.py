"""Synthetic CPU handoff checks; no detector, real GT or accuracy claim."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def rejected(call):
    try:call()
    except AssertionError:return
    raise RuntimeError('Invalid input was accepted')


def run():
    worker=load('evaluate_fog_cpu');runner=load('run_fog_cpu_evaluation_colab')
    import numpy as np
    class Boxes:
        def __init__(self):self.values={'token':[{'velocity':np.array([np.nan,1.]),'ego_translation':np.array([1.,2.,3.]),'score':np.float32(.5),'label':np.int64(1)}]}
        def serialize(self):return self.values
    boxes=Boxes();original=boxes.values['token'][0]['ego_translation'].copy()
    fingerprint=worker.gt_fingerprint(boxes)
    assert worker.gt_fingerprint(boxes)==fingerprint
    assert np.array_equal(boxes.values['token'][0]['ego_translation'],original)
    boxes.values['token'][0]['ego_translation'][0]=2.
    assert worker.gt_fingerprint(boxes)!=fingerprint
    data={'meta':{'use_lidar':True},'results':{'b':[{'sample_token':'b'}],'a':[]}}
    before=copy.deepcopy(data);padded=worker.pad_prediction(data,['a','b','c'],['a','b'])
    assert list(padded['results'])==['b','a','c'] and padded['results']['c']==[] and data==before
    rejected(lambda:worker.pad_prediction(data,['a','c'],['a','b']))
    rejected(lambda:worker.pad_prediction(data,['a','b','c'],['a','a']))
    frame={'sample_token':'a','index':0,'packed_points_sha256':'x','boxes_lidar':[[0.]*9],'scores':[.9],'labels':[0]}
    ledger={'packed_points_sha256':'x','prediction_count':1}
    runner.validate_frame(frame,'a',0,ledger)
    for key,value in [('scores',[1.1]),('labels',[1.5]),('boxes_lidar',[[float('nan')]*9])]:
        bad={**frame,key:value};rejected(lambda:runner.validate_frame(bad,'a',0,ledger))
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);archive=root/'example.zip';raw=b'verified'
        (root/'unchanged').write_bytes(raw);(root/'derived').write_bytes(b'derived')
        prior={'source_files':{'unchanged':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'derived':{'bytes':999,'sha256':'source-digest'}},'derived_sha256':{'derived':hashlib.sha256(b'derived').hexdigest()}}
        assert len(runner.verify_reusable_metadata(root,prior))==2
        (root/'derived').write_bytes(b'tampered')
        rejected(lambda:runner.verify_reusable_metadata(root,prior))
        manifest={'files':{'test.txt':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
        with zipfile.ZipFile(archive,'w') as z:z.writestr('test.txt',raw);z.writestr('manifest.json',json.dumps(manifest))
        assert runner.verified_archive(archive,runner.sha(archive))==manifest['files']
        rejected(lambda:runner.verified_archive(archive,'0'*64))
        with zipfile.ZipFile(archive,'a') as z:z.writestr('extra.txt',b'bad')
        rejected(lambda:runner.verified_archive(archive,runner.sha(archive)))
        classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
        for model in ('lidar','lidar_camera'):
            for level in range(4):
                folder=root/f'level_{level}_{model}';folder.mkdir()
                score=.6-.1*level+(.03 if model=='lidar_camera' else 0)
                summary={'mean_ap':score,'nd_score':score+.05,'mean_dist_aps':{name:score for name in classes},'tp_errors':{'trans_err':.1,'scale_err':.2,'orient_err':.3,'vel_err':.4,'attr_err':.1}}
                coverage={'samples':600,'filtered_gt_counts':{name:10 for name in classes},'filtered_prediction_counts':{name:20 for name in classes}}
                (folder/'metrics_summary.json').write_text(json.dumps(summary));(folder/'coverage.json').write_text(json.dumps(coverage))
        worker.report(root)
        import csv
        with (root/'fog_metrics.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==8 and abs(float(rows[3]['mAP_change_from_own_baseline_points'])+30)<1e-10
        with (root/'fog_class_metrics.csv').open() as f:assert len(list(csv.DictReader(f)))==80
        assert (root/'fog_severity_comparison.png').stat().st_size>1000
        assert (root/'fog_severity_comparison.pdf').stat().st_size>1000
    for name in ('evaluate_fog_cpu.py','run_fog_cpu_evaluation_colab.py'):compile((HERE/name).read_text(),str(HERE/name),'exec')
    return {'status':'FOG_CPU_EVALUATION_HELPER_CHECKS_PASS','checks':['GT fingerprint supports NumPy arrays/scalars and valid NaN velocities without mutation; detects changed values','metadata reuse verifies all unchanged/derived file hashes and rejects tampering','constructor-only padding preserves input order/content and rejects invalid populations','raw prediction validation rejects NaNs, invalid scores and noninteger classes','archive hash/member roundtrip and rejection','synthetic eight-row/80-class report, own-baseline changes, PNG/PDF rendering','Python syntax'],
        'worker_sha256':hashlib.sha256((HERE/'evaluate_fog_cpu.py').read_bytes()).hexdigest(),
        'coordinator_sha256':hashlib.sha256((HERE/'run_fog_cpu_evaluation_colab.py').read_bytes()).hexdigest(),
        'real_CPU_evaluation_performed':False,'scope':'Synthetic handoff checks; real SDK input/source checks and real accuracy execution remain in Colab.'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
