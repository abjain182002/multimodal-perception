"""Verify and analyse the frozen additional night/rain pedestrian input ZIP.

CPU only. No inference or official AP/NDS evaluation.
python analyse_case.py --archive source_inputs.zip --output results
"""
import argparse
import csv
import hashlib
import json
import platform
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from PIL import Image
import PIL

ARCHIVE_SHA256 = '1dcdef345a25febf7352cc393e06629e65c7935d4b4fd0fa3a47cd46b3a8ca97'
MAPPING = {'movable_object.barrier': 'barrier', 'vehicle.bicycle': 'bicycle',
           'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.car': 'car',
           'vehicle.construction': 'construction_vehicle', 'vehicle.motorcycle': 'motorcycle',
           'movable_object.trafficcone': 'traffic_cone', 'vehicle.trailer': 'trailer', 'vehicle.truck': 'truck',
           **{'human.pedestrian.' + x: 'pedestrian' for x in ['adult', 'child', 'construction_worker', 'police_officer']}}
RANGES = {**{k: 50 for k in ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']},
          **{k: 40 for k in ['pedestrian', 'bicycle', 'motorcycle']}, 'traffic_cone': 30, 'barrier': 30}
THRESHOLDS = (0.10, 0.30, 0.50)


def rotation(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def inside(box, point):
    local = rotation(box['rotation']).T @ (np.asarray(point) - box['translation'])
    w, l, h = box['size']
    return bool(np.all(np.abs(local) <= np.array([l, w, h]) / 2))


def filter_boxes(boxes, ego, racks, gt=False):
    out = []; counts = {'original': len(boxes), 'range_removed': 0, 'zero_points_removed': 0, 'rack_removed': 0}
    for box in boxes:
        if np.linalg.norm(np.asarray(box['translation'][:2]) - ego[:2]) >= RANGES[box['class']]:
            counts['range_removed'] += 1; continue
        if gt and box['num_lidar_pts'] + box['num_radar_pts'] == 0:
            counts['zero_points_removed'] += 1; continue
        if box['class'] in ('bicycle', 'motorcycle') and any(inside(r, box['translation']) for r in racks):
            counts['rack_removed'] += 1; continue
        out.append(box)
    return out, counts


def match(gt, predictions, threshold):
    used = set(); hits = []; unmatched = []
    for pi in sorted(range(len(predictions)), key=lambda i: (-predictions[i]['score'], i)):
        box = predictions[pi]
        if box['score'] < threshold: continue
        distance, gi = min(((float(np.linalg.norm(np.asarray(box['translation'][:2]) - g['translation'][:2])), j)
                            for j, g in enumerate(gt) if j not in used and g['class'] == box['class']),
                           default=(float('inf'), -1))
        if distance < 2: used.add(gi); hits.append((pi, gi))
        else: unmatched.append(pi)
    return hits, unmatched, [j for j in range(len(gt)) if j not in used]


def project(box, sensor, poses, calibrations):
    w, l, h = box['size']
    xyz = np.array([[x*l/2, y*w/2, z*h/2] for z in (-1, 1)
                    for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))]).T
    xyz = rotation(box['rotation']) @ xyz + np.asarray(box['translation'])[:, None]
    for pose in (poses[sensor['ego_pose_token']], calibrations[sensor['calibrated_sensor_token']]):
        xyz = rotation(pose['rotation']).T @ (xyz - np.asarray(pose['translation'])[:, None])
    if xyz[2].min() <= 0.2: return None
    uv = np.asarray(calibrations[sensor['calibrated_sensor_token']]['camera_intrinsic']) @ xyz
    return uv[:2] / uv[2]


def draw_box(ax, uv, colour, linestyle='-'):
    if uv is None: return
    edges = [(i, (i//4)*4 + (i+1) % 4) for i in range(8)] + [(i, i+4) for i in range(4)]
    for j, k in edges: ax.plot(uv[0, [j, k]], uv[1, [j, k]], color=colour, ls=linestyle, lw=1.6)


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def analyse(archive, output):
    archive_data = archive.read_bytes()
    assert hashlib.sha256(archive_data).hexdigest() == ARCHIVE_SHA256
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='positive_night_rain_') as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            manifest = json.loads(z.read('manifest.json'))['members']
            assert len(manifest) == 18 and set(z.namelist()) == set(manifest) | {'manifest.json'}
            for name, expected in manifest.items():
                relative = Path(name); assert not relative.is_absolute() and '..' not in relative.parts
                data = z.read(name)
                assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        read = lambda name: json.loads((root / name).read_text())
        selection, metadata = read('frozen_selection.json'), read('case_metadata_and_night_candidates.json')
        token = selection['sample_token']; audit = read('selection_audit.json')
        assert audit['prediction_files_opened'] is False and audit['selected'] == selection
        eligible = [r for r in audit['checked_frames'] if r['eligible']]
        chosen_scene = min(r['scene'] for r in eligible)
        chosen_frame = min((r for r in eligible if r['scene'] == chosen_scene),
                           key=lambda r: (r['distance_from_middle_timestamp'], r['sample_token']))
        assert chosen_frame['sample_token'] == token and chosen_scene == selection['scene_name']
        provenance = read('source_verification.json')
        assert provenance['protocol_sha256'] == hashlib.sha256((root/'selection_protocol.json').read_bytes()).hexdigest()
        assert provenance['frozen_selection_sha256'] == hashlib.sha256((root/'frozen_selection.json').read_bytes()).hexdigest()
        sd = {r['filename']: r for r in read('sample_data.json')}
        poses = {r['token']: r for r in read('ego_pose.json')}
        calibrations = {r['token']: r for r in read('calibrated_sensor.json')}
        inv = selection['sensor_inventory']; assert len(inv['lidar_files']) == 1
        lidar_file = inv['lidar_files'][0]; lidar_sd = sd[lidar_file]
        assert lidar_sd['sample_token'] == token and lidar_sd['is_key_frame'] is True
        ego = np.asarray(poses[lidar_sd['ego_pose_token']]['translation'])
        annotations = metadata['annotations']; assert all(a['sample_token'] == token for a in annotations)
        racks = [a for a in annotations if a['category_name'] == 'static_object.bicycle_rack']
        gt, gt_counts = filter_boxes([dict(a, **{'class': MAPPING[a['category_name']]}) for a in annotations if a['category_name'] in MAPPING], ego, racks, True)
        # Select camera and focal object using GT alone, without loading predictions.
        focal_options = []
        for gi, g in enumerate(gt):
            if g['class'] != 'pedestrian': continue
            views = []
            for filename in inv['camera_files']:
                assert sd[filename]['sample_token'] == token and sd[filename]['is_key_frame'] is True
                uv = project(g, sd[filename], poses, calibrations)
                if uv is None: continue
                with Image.open(root/'dataset'/filename) as image: size = image.size
                extent = np.maximum(0, np.minimum(uv.max(1), size) - np.maximum(uv.min(1), 0))
                if extent.prod() > 0: views.append((float(extent.prod()), filename, uv))
            if views:
                area, filename, uv = max(views, key=lambda v: (v[0], v[1]))
                distance = float(np.linalg.norm(np.asarray(g['translation'][:2]) - ego[:2]))
                focal_options.append((distance, g['token'], gi, filename, uv, area))
        assert focal_options, 'No eligible pedestrian has a valid camera projection; stop without replacing frame'
        distance, _, gi, filename, gt_uv, area = min(focal_options, key=lambda x: x[:2])
        g = gt[gi]
        assert g['token'] in {p['annotation_token'] for p in selection['eligible_pedestrians']}
        image = Image.open(root/'dataset'/filename).convert('RGB')
        lo, hi = gt_uv.min(1), gt_uv.max(1); centre = (lo + hi) / 2
        height = min(max(float(max(hi-lo))*2.5, 150), image.height); width = min(height*1.6, image.width)
        cx = np.clip(centre[0], width/2, image.width-width/2); cy = np.clip(centre[1], height/2, image.height-height/2)
        crop = (float(cx-width/2), float(cx+width/2), float(cy+height/2), float(cy-height/2))
        focal = {'scene': selection['scene_name'], 'sample_token': token, 'annotation_token': g['token'],
                 'camera': filename.split('/')[1], 'ego_xy_distance_m': distance,
                 'visibility_token': g['visibility_token'], 'num_lidar_pts': g['num_lidar_pts'], 'num_radar_pts': g['num_radar_pts'],
                 'projected_box_extent_px': (hi-lo).tolist(), 'clipped_projection_area_px2': area, 'crop_xy_limits': crop,
                 'rule': 'Nearest eligible projectable GT pedestrian, then largest clipped GT projection; no prediction access.'}
        dump(output/'focal_selection.json', focal)
        # Raw current-scan support cross-check, distinct from the multi-sweep model input.
        xyz = np.fromfile(root/'dataset'/lidar_file, dtype=np.float32).reshape(-1, 5)[:, :3].T
        cs = calibrations[lidar_sd['calibrated_sensor_token']]; ep = poses[lidar_sd['ego_pose_token']]
        world = rotation(ep['rotation']) @ (rotation(cs['rotation']) @ xyz + np.asarray(cs['translation'])[:, None]) + np.asarray(ep['translation'])[:, None]
        local = rotation(g['rotation']).T @ (world - np.asarray(g['translation'])[:, None])
        w, l, h = g['size']; measured_points = int(np.all(np.abs(local) <= np.asarray([l, w, h])[:, None]/2, axis=0).sum())
        assert measured_points == g['num_lidar_pts']
        frame_rows = []; focal_rows = []; details = []; view_boxes = []
        for model in ('lidar', 'lidar_camera'):
            saved = read('predictions/'+model+'.json'); assert set(saved['results']) == {token}
            predictions, pred_counts = filter_boxes([dict(b, score=b['detection_score'], **{'class': b['detection_name']}) for b in saved['results'][token]], ego, racks)
            same = [i for i, p in enumerate(predictions) if p['class'] == 'pedestrian']
            nearest = min(same, key=lambda i: (float(np.linalg.norm(np.asarray(predictions[i]['translation'][:2])-g['translation'][:2])), i)) if same else None
            within_two = [i for i in same if np.linalg.norm(np.asarray(predictions[i]['translation'][:2])-g['translation'][:2]) < 2]
            for threshold in THRESHOLDS:
                hits, fp, fn = match(gt, predictions, threshold)
                assigned = next((i for i, j in hits if j == gi), None)
                err = float(np.linalg.norm(np.asarray(predictions[nearest]['translation'][:2])-g['translation'][:2])) if nearest is not None else None
                frame_rows.append({'scene': selection['scene_name'], 'sample_token': token, 'model': model, 'threshold': threshold,
                                   'gt': len(gt), 'tp': len(hits), 'fp': len(fp), 'fn': len(fn),
                                   'precision': len(hits)/(len(hits)+len(fp)) if hits or fp else '', 'recall': len(hits)/len(gt),
                                   'scope': 'Ten-class local XY <2 m diagnostic; not AP/NDS'})
                focal_rows.append({'scene': selection['scene_name'], 'annotation_token': g['token'], 'model': model, 'threshold': threshold,
                                   'nearest_score': predictions[nearest]['score'] if nearest is not None else '', 'nearest_xy_error_m': err if err is not None else '',
                                   'same_class_candidates_within_2m': len(within_two),
                                   'maximum_score_within_2m': max((predictions[i]['score'] for i in within_two), default=''),
                                   'score_eligible_candidates_within_2m': sum(predictions[i]['score'] >= threshold for i in within_two),
                                   'matched': assigned is not None, 'assigned_score': predictions[assigned]['score'] if assigned is not None else '',
                                   'nearest_is_assigned': nearest == assigned if assigned is not None else False})
                details.append({'model': model, 'threshold': threshold, 'gt_filter': gt_counts, 'prediction_filter': pred_counts,
                                'matches': [{'prediction_index': i, 'annotation_token': gt[j]['token'], 'class': gt[j]['class'], 'score': predictions[i]['score']} for i, j in hits],
                                'unmatched_prediction_indices': fp, 'missed_annotation_tokens': [gt[j]['token'] for j in fn],
                                'focal_nearest_index': nearest, 'focal_assigned_index': assigned})
                if threshold == 0.30:
                    view_boxes.append({'model': model, 'uv': project(predictions[nearest], sd[filename], poses, calibrations) if nearest is not None else None,
                                       'matched': assigned is not None, 'nearest_assigned': nearest == assigned if assigned is not None else False,
                                       'score': predictions[nearest]['score'] if nearest is not None else None, 'error': err, 'counts': (len(hits),len(fp),len(fn))})
        write_csv(output/'frame_threshold_diagnostics.csv', frame_rows)
        write_csv(output/'pedestrian_threshold_diagnostics.csv', focal_rows)
        dump(output/'matching_details.json', details)
        # Original image values; fixed shared view/crop; full-view context above crop.
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout='constrained')
        for col, box in enumerate(view_boxes):
            for row in (0, 1):
                ax = axes[row, col]; ax.imshow(image)
                draw_box(ax, gt_uv, '#ff35a0')
                draw_box(ax, box['uv'], '#00d9ff', '-' if box['nearest_assigned'] else '--')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(('LiDAR-only' if col == 0 else 'Camera–LiDAR fusion') + ' | ' + focal['camera'], fontsize=12)
                    ax.set_xlim(0, image.width); ax.set_ylim(image.height, 0)
                    ax.plot([crop[0],crop[1],crop[1],crop[0],crop[0]], [crop[3],crop[3],crop[2],crop[2],crop[3]], color='#ffdd55', lw=1)
                else:
                    ax.set_xlim(crop[:2]); ax.set_ylim(crop[2:])
                    text = f"score {box['score']:.3f} | XY error {box['error']:.3f} m" if box['score'] is not None else 'No same-class candidate'
                    ax.set_xlabel(text + f"\n{'MATCH' if box['matched'] else 'MISS'} at 0.30 | full-frame TP/FP/FN {box['counts'][0]}/{box['counts'][1]}/{box['counts'][2]}", fontsize=10)
        fig.suptitle(f"{selection['scene_name']}: additional annotated pedestrian in a night/rain-labelled scene\nPink: ground truth | cyan: nearest candidate, dashed when not assigned at 0.30", fontsize=13)
        for ext in ('png', 'pdf'): fig.savefig(output/f'pedestrian_comparison.{ext}', dpi=180)
        plt.close(fig)
        report = {'status': 'ADDITIONAL_POSITIVE_NIGHT_RAIN_DIAGNOSTICS_PASS', 'archive_sha256': ARCHIVE_SHA256,
                  'verified_payload_members': 18, 'scene': selection['scene_name'], 'sample_token': token,
                  'filtered_gt_count': len(gt), 'eligible_pedestrians': sum(b['class']=='pedestrian' for b in gt),
                  'selection_rule_replayed_from_preserved_candidate_audit': True, 'focal_selection_before_prediction_access': True,
                  'current_scan_points_recomputed': measured_points, 'annotated_current_scan_points': g['num_lidar_pts'],
                  'threshold_rows_per_table': len(frame_rows), 'python': platform.python_version(),
                  'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__, 'Pillow': PIL.__version__,
                  'scope': 'One additional preselected positive frame; local filtering/matching and point-support check. No inference, official AP/NDS or physical weather-causality validation. Derived Drive copy pending.'}
        dump(output/'analysis_audit.json', report)
        print(json.dumps({'focal': focal, 'primary_focal': [r for r in focal_rows if r['threshold']==0.30],
                          'primary_frame': [r for r in frame_rows if r['threshold']==0.30], 'audit': report}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); analyse(args.archive, args.output)
