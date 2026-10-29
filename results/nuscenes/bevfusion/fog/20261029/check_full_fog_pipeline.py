"""CPU checks for cache I/O, annotation invariance and artifact recovery helpers.

The adapter's numerical source equivalence and GPU pilot are separate evidence.
These tests do not execute the full detector study.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def run():
    worker = module('full_worker', HERE / 'fog_full_worker.py')
    runner = module('full_runner', HERE / 'run_fog_full_study_colab.py')
    original = module('original', HERE / 'fog_gpu_smoke.py')
    class IdentityAdapter:
        def camera(self, rgb, level, relative): return rgb.copy()
        def points(self, points, level, relative): return points.copy(), {'fog_responses': 0}
    worker._ADAPTER = IdentityAdapter()
    completed = []
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp); worker.RAW_ROOT = temp / 'raw'; worker.RAW_ROOT.mkdir()
        image = worker.RAW_ROOT / 'samples/CAM_FRONT/test.jpg'; image.parent.mkdir(parents=True)
        assert cv2.imwrite(str(image), np.arange(12*16*3, dtype=np.uint8).reshape(12,16,3))
        scan = worker.RAW_ROOT / 'samples/LIDAR_TOP/test.pcd.bin'; scan.parent.mkdir(parents=True)
        points = np.arange(25, dtype=np.float32).reshape(5,5); points.tofile(scan)
        outputs = {}
        for path in (image, scan):
            relative = str(path.relative_to(worker.RAW_ROOT)); record = {'bytes':path.stat().st_size,'sha256':worker.sha(path)}
            task = (relative,record,0,str(temp/'cache'))
            name,saved = worker.generate(task); outputs[name] = saved
            assert worker.generate(task) == (name,saved)
            bad = {**record,'sha256':'0'*64}
            try: worker.generate((relative,bad,0,str(temp/'bad'))); raise RuntimeError('Wrong input hash accepted')
            except AssertionError: pass
        completed.append('baseline PNG/raw cache roundtrip, reusable cache and wrong source hash rejection')
        lidar = 'samples/LIDAR_TOP/test.pcd.bin'; camera = 'samples/CAM_FRONT/test.jpg'
        row = {'token':'a'*32,'images':{'CAM_FRONT':{'img_path':camera,'cam2img':np.eye(3)}},
            'lidar_points':{'lidar_path':lidar,'lidar2ego':np.eye(4)},
            'lidar_sweeps':[{'lidar_points':{'lidar_path':lidar,'timestamp':1}}],
            'instances':[{'bbox_3d':np.array([1.,np.nan,3.])}], 'ego2global':np.eye(4)}
        source = {'metainfo':{'classes':['car']},'data_list':[row,copy.deepcopy(row)]}
        result = worker.rewrite_annotation(source,outputs,original)
        assert result['data_list'][0]['images']['CAM_FRONT']['img_path'] == outputs[camera]['path']
        assert source['data_list'][0]['images']['CAM_FRONT']['img_path'] == camera
        for new,old in zip(result['data_list'],source['data_list']):
            for (c,k),(o,ok) in zip(original.path_fields(new),original.path_fields(old)): c[k]=o[ok]
        original.same(result,source)
        completed.append('multi-row annotation path-only rewrite, NaNs/calibration preserved and source immutable')
        artifact = temp / 'artifact'; artifact.mkdir(); (artifact/'record.json').write_text('{"valid":true}')
        (artifact/'ignore.partial').write_text('unfinished')
        archive = temp/'artifact.zip'; receipt = runner.archive_folder(artifact,archive)
        records = runner.check_zip(archive,receipt['sha256']); assert set(records)=={'record.json'}
        with zipfile.ZipFile(archive,'a') as z: z.writestr('extra.txt','bad')
        try: runner.check_zip(archive,runner.sha(archive)); raise RuntimeError('Extra artifact member accepted')
        except AssertionError: pass
        completed.append('immutable artifact roundtrip and partial-file exclusion; extra member rejected')
        runner.WORK = temp/'work'; (runner.WORK/'restoration').mkdir(parents=True)
        tokens = [f'{i:032x}' for i in range(600)]
        (runner.WORK/'restoration/sample_tokens.txt').write_text('\n'.join(tokens)+'\n')
        level_dest = temp/'drive'; level_dest.mkdir(); summary = temp/'summary'; summary.mkdir()
        protocol = {'fixed':True}
        for model in ('lidar','lidar_camera'):
            folder = temp/model; folder.mkdir()
            ledger = [{'sample_token':t,'packed_points_sha256':'1'*64} for t in tokens]
            worker.atomic(folder/'input_checks.json',ledger); worker.atomic(folder/'run_protocol.json',protocol)
            worker.atomic(folder/'results_nusc.json',{'results':{t:[] for t in tokens}})
            audit={'status':'FOG_FULL_MODEL_CONDITION_INFERENCE_PASS','model':model,'level':0,'samples':600,
                'worker_sha256':runner.WORKER_SHA,'strict_determinism':True,'accepted_pilot_prediction_match':True,
                'results_sha256':runner.sha(folder/'results_nusc.json'),'input_checks_sha256':runner.sha(folder/'input_checks.json')}
            worker.atomic(folder/'inference_audit.json',audit); runner.preserve_model(folder,level_dest)
        runner.collect_pair(level_dest,protocol,summary,0)
        assert json.loads((summary/'level_0_paired_check.json').read_text())['same_packed_points_all_frames']
        try: runner.collect_pair(level_dest,{'different':True},summary,0); raise RuntimeError('Wrong protocol accepted')
        except AssertionError:pass
        completed.append('600-token completed-artifact recovery/pair check; changed protocol rejected')
    for name in ('fog_full_worker.py','run_fog_full_study_colab.py'):
        compile((HERE/name).read_text(),str(HERE/name),'exec')
    result={'status':'FULL_FOG_PIPELINE_CPU_HELPER_CHECKS_PASS','checks':completed+['both scripts compile'],
        'worker_sha256':hashlib.sha256((HERE/'fog_full_worker.py').read_bytes()).hexdigest(),
        'coordinator_sha256':hashlib.sha256((HERE/'run_fog_full_study_colab.py').read_bytes()).hexdigest(),
        'scope':'Synthetic helper checks only; no full-study GPU execution or new fog numerical equivalence claim.'}
    return result


if __name__ == '__main__':
    result=run(); print(json.dumps(result,indent=2))
