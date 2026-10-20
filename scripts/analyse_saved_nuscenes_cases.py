"""CPU-only diagnostic replay from the preserved six-case input archive.

Run: python scripts/analyse_saved_nuscenes_cases.py --inputs INPUT.zip --output OUT
No inference, SDK filtering or mAP/NDS evaluation is performed here.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

ARCHIVE_SHA = '574a5ea93c4fc455317562caaa82a30f25ce3f8b20715ddbf2e05c0cff0b9bdd'
HIGHLIGHTS = {
    'b92ee1ed90c242d18bce9a8fdbbeb8d8': [22, 33, 31],
    '0f483d8f22694132946c831f5d644cd7': [26, 36, 34],
}


def distance(a, b):
    return math.hypot(a['translation'][0] - b['translation'][0],
                      a['translation'][1] - b['translation'][1])


def match(gt, predictions, threshold):
    available = set(range(len(gt)))
    pairs, fp = [], []
    eligible = [i for i, p in enumerate(predictions) if p['score'] >= threshold]
    for pi in sorted(eligible, key=lambda i: (-predictions[i]['score'], i)):
        pred = predictions[pi]
        candidates = [(distance(pred, gt[gi]), gi) for gi in available
                      if gt[gi]['class'] == pred['class']]
        d, gi = min(candidates, default=(math.inf, -1))
        if d < 2.0:
            pairs.append({'prediction_index': pi, 'ground_truth_index': gi,
                          'distance_metres': d})
            available.remove(gi)
        else:
            fp.append(pi)
    return {'matches': pairs, 'false_positive_indices': fp,
            'false_negative_indices': sorted(available)}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA, 'Input archive hash mismatch'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        def read(name):
            return json.loads(archive.read(name))
        contexts = read('candidate_contexts.json')
        selection = read('selection.json')['cases']
        candidates = {r['sample_token']: r for group in read('case_candidates.json').values() for r in group}
        input_hashes = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in
                        ['candidate_contexts.json', 'selection.json', 'selection_protocol.json', 'case_candidates.json']}
    args.output.mkdir(parents=True, exist_ok=True)
    summary, objects, classes, pairs_out, discordant = [], [], [], [], []
    replayed = 0
    for token in selection:
        context = contexts[token]
        gt = context['ground_truth']
        scene = candidates[token]['scene_name']
        for threshold in [0.1, 0.3, 0.5]:
            result = {m: match(gt, p, threshold) for m, p in context['predictions'].items()}
            sets = {m: {x['ground_truth_index'] for x in r['matches']} for m, r in result.items()}
            for model, r in result.items():
                tp, fp, fn = len(r['matches']), len(r['false_positive_indices']), len(r['false_negative_indices'])
                assert tp + fn == len(gt)
                if threshold == 0.3:
                    old = context['matching'][model]
                    assert r['matches'] == old['matches'], (scene, model, 'matches')
                    assert r['false_positive_indices'] == old['false_positive_indices']
                    assert r['false_negative_indices'] == old['false_negative_indices']
                    replayed += 1
                summary.append(dict(scene=scene, sample_token=token, threshold=threshold, model=model,
                                    gt=len(gt), tp=tp, fp=fp, fn=fn,
                                    precision=tp/(tp+fp) if tp+fp else None,
                                    recall=tp/len(gt),
                                    matched_xy_median_m=statistics.median(x['distance_metres'] for x in r['matches']) if tp else None,
                                    fusion_only=len(sets['lidar_camera']-sets['lidar']),
                                    lidar_only=len(sets['lidar']-sets['lidar_camera']),
                                    shared_matches=len(sets['lidar']&sets['lidar_camera']),
                                    shared_misses=len(set(range(len(gt)))-sets['lidar']-sets['lidar_camera'])))
            if threshold != 0.3 or token not in HIGHLIGHTS:
                continue
            for cl in sorted({b['class'] for b in gt}):
                indices = {i for i,b in enumerate(gt) if b['class']==cl}
                classes.append(dict(scene=scene, class_name=cl, gt=len(indices),
                                    lidar_tp=len(indices&sets['lidar']), fusion_tp=len(indices&sets['lidar_camera']),
                                    fusion_only=len(indices&(sets['lidar_camera']-sets['lidar'])),
                                    lidar_only=len(indices&(sets['lidar']-sets['lidar_camera']))))
            maps = {m: {x['ground_truth_index']: x for x in r['matches']} for m,r in result.items()}
            for gi in sorted(sets['lidar']&sets['lidar_camera']):
                ld=maps['lidar'][gi]['distance_metres']; fd=maps['lidar_camera'][gi]['distance_metres']
                pairs_out.append(dict(scene=scene, sample_token=token, gt_index=gi, class_name=gt[gi]['class'],
                                      lidar_distance_m=ld, fusion_distance_m=fd, fusion_minus_lidar_m=fd-ld))
            for gi in sorted(sets['lidar'] ^ sets['lidar_camera']):
                for model, predictions in context['predictions'].items():
                    nearest = sorted((distance(p,gt[gi]), i) for i,p in enumerate(predictions) if p['class']==gt[gi]['class'])
                    d, pi = nearest[0] if nearest else (None,None)
                    actual = maps[model].get(gi)
                    p = predictions[pi] if pi is not None else None
                    record = dict(scene=scene, sample_token=token, gt_index=gi, class_name=gt[gi]['class'], model=model,
                                        matched=actual is not None,
                                        assigned_prediction_index=actual['prediction_index'] if actual else None,
                                        assigned_score=predictions[actual['prediction_index']]['score'] if actual else None,
                                        assigned_distance_m=actual['distance_metres'] if actual else None,
                                        nearest_same_class_index=pi, nearest_same_class_distance_m=d,
                                        nearest_same_class_score=p['score'] if p else None,
                                        same_class_within_2m=sum(distance(q,gt[gi])<2 for q in predictions if q['class']==gt[gi]['class']),
                                        eligible_same_class_within_2m=sum(distance(q,gt[gi])<2 and q['score']>=.3 for q in predictions if q['class']==gt[gi]['class']),
                                        eligible_nearby_assignments=json.dumps([
                                            {'prediction_index': x['prediction_index'], 'assigned_gt_index': x['ground_truth_index']}
                                            for x in result[model]['matches']
                                            if predictions[x['prediction_index']]['class']==gt[gi]['class']
                                            and distance(predictions[x['prediction_index']],gt[gi])<2],separators=(',',':')))
                    for other_threshold in [.1,.5]:
                        other = match(gt, predictions, other_threshold)
                        record[f'matched_at_{other_threshold}'] = any(x['ground_truth_index']==gi for x in other['matches'])
                    discordant.append(record)
                    if gi in HIGHLIGHTS[token]:
                        objects.append(record)
    for name, rows in [('threshold_sensitivity.csv',summary),('highlighted_objects.csv',objects),
                       ('class_breakdown.csv',classes),('shared_match_localisation.csv',pairs_out),
                       ('discordant_objects.csv',discordant)]:
        write_csv(args.output/name,rows)
    audit = dict(status='SAVED_CASE_DIAGNOSTIC_REPLAY_PASS', input_archive_sha256=ARCHIVE_SHA,
                 input_hashes=input_hashes, scenes=len(selection), thresholds=[.1,.3,.5],
                 saved_model_case_matches_reproduced_exactly=replayed, summary_rows=len(summary),
                 highlight_rows=len(objects), source_prediction_hashes=read_protocol_hashes(raw),
                 scope='Replayed matching on six preserved, already SDK-filtered case contexts. No new inference, fresh SDK filtering, full-split evaluation, confidence intervals or causal tests.')
    (args.output/'replay_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


def read_protocol_hashes(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return json.loads(archive.read('selection_protocol.json'))['prediction_hashes']


if __name__ == '__main__':
    main()
