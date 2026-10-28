"""Prepare one annotation-positive night/rain pedestrian case on CPU.

Run with Drive mounted. Select and preserve the frame before reading model
predictions. Existing two night cases remain unchanged. No inference.
"""
import codecs
import gc
import hashlib
import io
import json
import math
import re
import shutil
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
ORIGINAL_NAME = 'nuscenes_validation_parts_01_02_verified.zip'
KEYFRAMES_NAME = 'queue_v1.0-trainval10_keyframes_verified.zip'
SOURCE_HASHES = {
    ORIGINAL_NAME: 'c89bc342a240d0bde22ef8b9200ebaf6de5b5f91c8a399073c4057d866f39c3a',
    KEYFRAMES_NAME: 'c1645861b3da66d638387731830393f0e06185c8396719b3f58ec38118a6381f',
}
PRED_HASHES = {
    'lidar': '15667405cb1af580e6f8082dcc50366560e45335c36f7122ec490f377cd86c63',
    'lidar_camera': 'dcd86893985366881b27d3841b64b93a85988158b518f0c9edfa274c291596dd',
}
PEDESTRIANS = {'human.pedestrian.' + x for x in ('adult', 'child', 'construction_worker', 'police_officer')}
PROTOCOL = {
    'question': 'Add one frame with eligible annotated pedestrians to the existing illustrative night/rain case study.',
    'metadata_condition': 'Scene description matches night/nighttime AND rain/rainy/raining as words, case insensitive.',
    'population': 'Official validation frames whose six camera keyframes and current LiDAR keyframe are preserved in the hash-verified part-10 keyframe backup.',
    'pedestrian_eligibility': 'One of four official pedestrian categories; combined annotated current LiDAR/radar points >0; ego XY distance strictly <40 m at current LiDAR pose.',
    'scene_rule': 'First scene by (scene name, scene token) containing at least one eligible frame.',
    'frame_rule': 'Eligible frame nearest the middle timestamped validation frame of that scene; ties by sample token.',
    'focal_rule': 'For later visual analysis: nearest eligible pedestrian with a valid camera projection; annotation token breaks distance ties. Choose camera by largest clipped GT projection area, never model outcome.',
    'matching_plan': 'Retain prior ten-class local filters and score-ordered one-to-one XY <2 m matching at 0.10, 0.30 and 0.50; 0.30 primary. No threshold tuning.',
    'amendment': 'Keep original scene-1059 and scene-1060 results. This is an additional annotation-selected positive case, not replacement of a negative result.',
    'no_candidate_rule': 'Stop and report no eligible frame. Do not relax labels, range, support or archive availability automatically.',
    'prediction_access': 'Neither model prediction file is opened until protocol and annotation-based selection have been saved and verified on Drive.',
    'scope': 'Deterministic availability-restricted illustrative selection, not random sampling, representative weather accuracy or a causal night/rain comparison.',
}


def sha(path, progress=False):
    h = hashlib.sha256(); total = last = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block); total += len(block)
            if progress and total - last >= 2_000_000_000:
                print('Hash-checked GB:', round(total / 1e9, 2), flush=True); last = total
    return h.hexdigest()


def array_rows(stream):
    """Bounded-buffer UTF-8 JSON-array reader, including streaming TAR members."""
    text = codecs.getreader('utf-8')(stream)
    decoder = json.JSONDecoder(); buf = ''; pos = 0; started = False; done = False
    while not done:
        while True:
            while pos < len(buf) and buf[pos].isspace(): pos += 1
            if not started:
                if pos == len(buf): break
                assert buf[pos] == '['; pos += 1; started = True; continue
            while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ','): pos += 1
            if pos == len(buf): break
            if buf[pos] == ']': done = True; break
            try: row, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError: break
            yield row; pos = end
        if done: break
        chunk = text.read(1024 * 1024)
        if not chunk: raise ValueError('Incomplete JSON array')
        buf = buf[pos:] + chunk; pos = 0
    assert not (buf[pos + 1:] + text.read()).strip()


def tables(archive, selectors):
    out = {}
    with zipfile.ZipFile(archive) as z:
        with z.open('metadata/v1.0-trainval_meta.tgz') as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    name = Path(member.name).name
                    if member.isfile() and name in selectors:
                        print('STREAMING:', name, flush=True); chosen = []
                        for index, row in enumerate(array_rows(tar.extractfile(member)), 1):
                            if selectors[name](row): chosen.append(row)
                            if index % 400000 == 0: print(name, 'records scanned:', index, flush=True)
                        out[name] = chosen
                        print('SELECTED:', name, len(chosen), flush=True)
                        if set(out) == set(selectors): break
    assert set(out) == set(selectors), set(selectors) - set(out)
    return out


def keyframes(inventory):
    cameras = inventory['camera_files']
    lidar = [x for x in inventory['lidar_files'] if x.startswith('samples/')]
    assert len(cameras) == len(set(cameras)) == 6 and len(lidar) == 1
    return set(cameras) | set(lidar)


def condition(description):
    return bool(re.search(r'\bnight(?:time)?\b', description, re.I) and
                re.search(r'\brain(?:y|ing)?\b', description, re.I))


def choose_frame(scene_rows, samples, inventory, annotations, sample_data, poses, available):
    """Pure metadata/annotation selection. Has no prediction parameter or access."""
    sd = {r['filename']: r for r in sample_data}
    ep = {r['token']: r for r in poses}
    by_sample = {}
    for a in annotations: by_sample.setdefault(a['sample_token'], []).append(a)
    inventory_by_token = {r['sample_token']: r for r in inventory}
    audit = []
    for scene in sorted(scene_rows, key=lambda s: (s['name'], s['token'])):
        frames = sorted([s for s in samples if s['scene_token'] == scene['token'] and s['token'] in inventory_by_token],
                        key=lambda s: (s['timestamp'], s['token']))
        if not frames: continue
        center_time = frames[len(frames) // 2]['timestamp']; eligible = []
        for frame in frames:
            token = frame['token']; inv = inventory_by_token[token]
            files = keyframes(inv)
            if not files <= available:
                audit.append({'scene': scene['name'], 'sample_token': token, 'eligible': False, 'reason': 'keyframes unavailable in frozen backup'})
                continue
            lidar = next(x for x in files if x.startswith('samples/LIDAR_TOP/'))
            ego = ep[sd[lidar]['ego_pose_token']]['translation']
            peds = []
            for a in by_sample.get(token, []):
                if a['category_name'] not in PEDESTRIANS: continue
                distance = math.hypot(a['translation'][0] - ego[0], a['translation'][1] - ego[1])
                if a['num_lidar_pts'] + a['num_radar_pts'] > 0 and distance < 40:
                    peds.append({'annotation_token': a['token'], 'ego_xy_distance_m': distance,
                                 'visibility_token': a['visibility_token'], 'num_lidar_pts': a['num_lidar_pts'],
                                 'num_radar_pts': a['num_radar_pts']})
            peds.sort(key=lambda p: (p['ego_xy_distance_m'], p['annotation_token']))
            entry = {'scene': scene['name'], 'sample_token': token, 'timestamp': frame['timestamp'],
                     'eligible': bool(peds), 'eligible_pedestrians': peds,
                     'distance_from_middle_timestamp': abs(frame['timestamp'] - center_time)}
            audit.append(entry)
            if peds: eligible.append((entry['distance_from_middle_timestamp'], token, frame, peds))
        if eligible:
            _, token, frame, peds = min(eligible, key=lambda x: (x[0], x[1]))
            return {'sample_token': token, 'scene_name': scene['name'], 'scene_description': scene['description'],
                    'scene_token': scene['token'], 'selection': PROTOCOL['scene_rule'] + ' ' + PROTOCOL['frame_rule'],
                    'sensor_inventory': inventory_by_token[token], 'eligible_pedestrians': peds}, audit
    return None, audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    path.write_bytes(data); assert path.read_bytes() == data


def main():
    started = time.monotonic()
    assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive first'
    for name in SOURCE_HASHES: assert (BASE / name).is_file(), f'Missing preserved archive: {name}'
    for model in PRED_HASHES:
        assert (BASE / 'full_validation_merged' / model / 'results_nusc.json').is_file(), f'Missing saved full predictions: {model}'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = BASE / 'night_rain_pedestrian_20261028' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'selection_protocol.json', PROTOCOL)
    protocol_hash = sha(destination / 'selection_protocol.json')
    print('ANNOTATION-BASED SELECTION PROTOCOL PRESERVED:', protocol_hash, flush=True)
    for name, expected in SOURCE_HASHES.items():
        print('VERIFYING SOURCE:', name, flush=True)
        assert sha(BASE / name, True) == expected, name
        print('SOURCE HASH PASS', flush=True)
    with zipfile.ZipFile(BASE / ORIGINAL_NAME) as z:
        inventory = json.loads(z.read('preparation/validation_input_manifest.json'))
    assert len(inventory) == len({r['sample_token'] for r in inventory}) == 6019
    validation_tokens = {r['sample_token'] for r in inventory}
    with zipfile.ZipFile(BASE / KEYFRAMES_NAME) as z:
        available = {n[len('dataset/'):] for n in z.namelist() if n.startswith('dataset/samples/') and not n.endswith('/')}

    first = tables(BASE / ORIGINAL_NAME, {name: (lambda r: True) for name in
                   ('scene.json', 'sample.json', 'category.json', 'instance.json', 'visibility.json')})
    samples = {s['token']: s for s in first['sample.json']}
    assert validation_tokens <= set(samples)
    val_scenes = {samples[t]['scene_token'] for t in validation_tokens}
    scenes = [s for s in first['scene.json'] if s['token'] in val_scenes and condition(s['description'])]
    scene_tokens = {s['token'] for s in scenes}
    candidate_tokens = {t for t in validation_tokens if samples[t]['scene_token'] in scene_tokens}
    assert scenes and candidate_tokens, 'No validation scene with explicit night and rain labels'
    cats = {c['token']: c['name'] for c in first['category.json']}
    instance_category = {i['token']: cats[i['category_token']] for i in first['instance.json']}
    candidates = [r for r in inventory if r['sample_token'] in candidate_tokens]
    available_candidate_files = set().union(*(keyframes(r) for r in candidates if keyframes(r) <= available))
    assert available_candidate_files, 'No candidate frame has all required keyframes in the frozen backup'
    second = tables(BASE / ORIGINAL_NAME, {
        'sample_annotation.json': lambda r: r['sample_token'] in candidate_tokens,
        'sample_data.json': lambda r: r['filename'] in available_candidate_files,
        'calibrated_sensor.json': lambda r: True,
    })
    annotations = [dict(a, category_name=instance_category[a['instance_token']]) for a in second['sample_annotation.json']]
    pose_tokens = {s['ego_pose_token'] for s in second['sample_data.json']}
    third = tables(BASE / ORIGINAL_NAME, {'ego_pose.json': lambda r: r['token'] in pose_tokens})
    assert {r['token'] for r in third['ego_pose.json']} == pose_tokens
    selected, selection_audit = choose_frame(scenes, list(samples.values()), candidates, annotations,
                                            second['sample_data.json'], third['ego_pose.json'], available)
    write_json(destination / 'selection_audit.json', {'protocol_sha256': protocol_hash, 'candidate_scenes': scenes,
                                                    'checked_frames': selection_audit, 'selected': selected,
                                                    'prediction_files_opened': False})
    assert selected is not None, 'No annotation-positive eligible pedestrian frame; audit saved. Stop without relaxing selection.'
    write_json(destination / 'frozen_selection.json', selected)
    selection_hash = sha(destination / 'frozen_selection.json')
    print('POSITIVE FRAME SELECTED BEFORE PREDICTIONS:', json.dumps(selected), flush=True)

    work = Path('/content/night_rain_pedestrian_transfer') / stamp
    work.mkdir(parents=True, exist_ok=False)
    token = selected['sample_token']; required = keyframes(selected['sensor_inventory'])
    sd = [s for s in second['sample_data.json'] if s['filename'] in required]
    assert len(sd) == 7 and {s['filename'] for s in sd} == required
    cs_tokens = {s['calibrated_sensor_token'] for s in sd}; ep_tokens = {s['ego_pose_token'] for s in sd}
    cs = [c for c in second['calibrated_sensor.json'] if c['token'] in cs_tokens]
    ep = [p for p in third['ego_pose.json'] if p['token'] in ep_tokens]
    assert {c['token'] for c in cs} == cs_tokens and {p['token'] for p in ep} == ep_tokens
    metadata = {'night_candidates': [selected], 'selected_samples': [samples[token]],
                'selected_scenes': [s for s in scenes if s['token'] == selected['scene_token']],
                'annotations': [a for a in annotations if a['sample_token'] == token],
                'visibility_definitions': first['visibility.json'], 'scope': PROTOCOL['scope']}
    write_json(work / 'case_metadata_and_night_candidates.json', metadata)
    write_json(work / 'sample_data.json', sd); write_json(work / 'calibrated_sensor.json', cs); write_json(work / 'ego_pose.json', ep)
    for name in ('selection_protocol.json', 'selection_audit.json', 'frozen_selection.json'):
        shutil.copy2(destination / name, work / name)
        assert sha(destination / name) == sha(work / name)
    with zipfile.ZipFile(BASE / KEYFRAMES_NAME) as z:
        for name in sorted(required):
            path = work / 'dataset' / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read('dataset/' + name))
    provenance = {'archive_sha256': SOURCE_HASHES, 'protocol_sha256': protocol_hash,
                  'frozen_selection_sha256': selection_hash, 'models': {},
                  'preparation_script_sha256': sha(Path(__file__)),
                  'preparation_script_url': globals().get('PREPARATION_SCRIPT_URL', 'not supplied')}
    for model, expected in PRED_HASHES.items():
        # This is the first point at which either prediction file is read.
        assert sha(destination / 'frozen_selection.json') == selection_hash
        path = BASE / 'full_validation_merged' / model / 'results_nusc.json'
        print('VERIFYING SAVED FULL PREDICTIONS:', model, flush=True)
        assert sha(path, True) == expected, model
        with path.open() as f: data = json.load(f)
        assert set(data['results']) == validation_tokens
        boxes = data['results'][token]; assert len(boxes) <= 500
        for b in boxes:
            assert b['sample_token'] == token
            assert all(math.isfinite(float(v)) for k in ('translation', 'size', 'rotation', 'velocity') for v in b[k])
            assert math.isfinite(float(b['detection_score']))
        write_json(work / 'predictions' / (model + '.json'), {'results': {token: boxes}, 'meta': data.get('meta', {})})
        provenance['models'][model] = {'full_results_sha256': expected, 'full_tokens': 6019, 'selected_tokens': [token]}
        del boxes, data; gc.collect()
        print('SAVED PREDICTION SUBSET VERIFIED:', model, flush=True)
    write_json(work / 'source_verification.json', provenance)
    manifest = {str(p.relative_to(work)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(work.rglob('*')) if p.is_file()}
    write_json(work / 'manifest.json', {'members': manifest, 'scope': 'One added annotation-selected positive night/rain frame; visual analysis and diagnostic matching pending.'})
    archive = work.parent / ('night_rain_pedestrian_inputs_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(work.rglob('*')):
            if path.is_file(): z.write(path, str(path.relative_to(work)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, entry in manifest.items():
            content = z.read(name)
            assert len(content) == entry['bytes'] and hashlib.sha256(content).hexdigest() == entry['sha256']
    target = destination / archive.name
    shutil.copy2(archive, target); digest = sha(archive); assert sha(target) == digest
    with zipfile.ZipFile(target) as z:
        for name, entry in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == entry['sha256'], name
    receipt = {'status': 'POSITIVE_NIGHT_RAIN_INPUTS_AND_DRIVE_COPY_PASS', 'archive': target.name,
               'bytes': target.stat().st_size, 'sha256': digest, 'sensor_files': 7,
               'selected_sample_token': token, 'scene': selected['scene_name'],
               'eligible_pedestrians': len(selected['eligible_pedestrians']), 'verified_payload_members': len(manifest),
               'elapsed_seconds': round(time.monotonic() - started, 1),
               'scope': 'Protocol, annotation-based selection, inputs and saved prediction subset verified. No inference. Visual review and matching remain pending.'}
    write_json(target.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True); print('UPLOAD THIS ZIP:', target, flush=True)
    from google.colab import files
    files.download(str(target))


if __name__ == '__main__':
    main()
