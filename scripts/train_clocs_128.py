"""128-frame internal development study. Candidate AP is NOT KITTI AP40."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

from fusion.clocs_network import CLOCsFusionNet
from fusion.clocs_dataset import training_example_content_sha256
from fusion.clocs_sampling import split_frames, select_candidates
from fusion.clocs_training import sigmoid_focal_loss
from scripts.train_clocs_pilot import load_frame, sha256, average_precision


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def predict(model, frame, device):
    return model(torch.from_numpy(frame['features']).to(device),
                 torch.from_numpy(frame['candidate_index']).to(device),
                 int(frame['example']['num_candidates']))['candidate_logits']


def evaluate(model, root, ids, device):
    model.eval()
    labels_all, base_all, fused_all, losses = [], [], [], []
    with torch.no_grad():
        for fid in ids:
            frame = load_frame(root, fid)
            labels = np.asarray(frame['example']['labels'])
            valid = labels >= 0
            logits = predict(model, frame, device)
            mask = torch.from_numpy(valid).to(device)
            targets = torch.from_numpy(np.asarray(frame['example']['targets'], dtype=np.float32)).to(device)
            losses.append(float(sigmoid_focal_loss(logits[mask], targets[mask], reduction='mean').item()))
            labels_all.append(labels[valid])
            base_all.append(np.asarray(frame['example']['candidate_scores_3d'])[valid])
            fused_all.append(torch.sigmoid(logits).cpu().numpy()[valid])
    labels = np.concatenate(labels_all)
    return dict(frames=len(ids), evaluated_candidates=len(labels), positive_candidates=int(labels.sum()),
                mean_frame_focal_loss=float(np.mean(losses)),
                pointpillars_candidate_ap=average_precision(labels, np.concatenate(base_all)),
                fusion_candidate_ap=average_precision(labels, np.concatenate(fused_all)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--batch-manifest', type=Path, required=True)
    parser.add_argument('--train-ids', type=Path, required=True)
    parser.add_argument('--val-ids', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--negative-ratio', type=int, choices=[0, 4, 8, 16], default=0)
    parser.add_argument('--background-cap', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.background_cap < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Invalid epochs, background cap or learning rate')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new output directory; existing results will not be overwritten')
    root = args.data_dir
    batch = json.loads(args.batch_manifest.read_text())
    summary_path = root / 'clocs_yolo26_train_128_summary.json'
    summary = json.loads(summary_path.read_text())
    ids = sorted(f['frame_id'] for f in batch['frames'])
    train_pool = set(args.train_ids.read_text().split())
    val_pool = set(args.val_ids.read_text().split())
    if (batch['split'] != 'train' or train_pool & val_pool or not set(ids) <= train_pool
            or sha256(args.train_ids) != batch['train_split_sha256']
            or sha256(args.val_ids) != batch['validation_split_sha256']):
        raise ValueError('Frozen dataset split verification failed')
    train_ids, dev_ids = split_frames(ids, args.seed)
    if summary['status'] != 'PASS' or summary['completed_frames'] != 128 or sorted(summary['frame_ids']) != ids:
        raise ValueError('Summary differs from batch manifest')
    patterns = [('examples', '*_real_targets.npz'), ('rows', '*_clocs_rows.npz'), ('audits', 'clocs_yolo26_frame_*_audit.json')]
    if any(len(list((root / folder).glob(pattern))) != 128 for folder, pattern in patterns):
        raise ValueError('Expected exactly 128 examples, rows and audits')
    input_hashes, positives = {}, {}
    total_keys = ('car_gt_after_range_filter', 'yolo26x_car_detections', 'num_candidates', 'num_feature_rows', 'num_positive', 'num_negative', 'num_ignored')
    totals = dict.fromkeys(total_keys, 0)
    for fid in ids:
        frame = load_frame(root, fid)
        audit_path = root / 'audits' / f'clocs_yolo26_frame_{fid}_audit.json'
        audit = json.loads(audit_path.read_text())
        example = frame['example']
        if frame['camera_model'] != 'yolo26x' or str(example['frame_id']) != fid or audit['frame_id'] != fid or audit['split'] != 'train':
            raise ValueError(f'Frame identity/camera mismatch: {fid}')
        if sha256(frame['rows_path']) != audit['source_sha256']['feature_rows']:
            raise ValueError(f'Rows hash differs from audit: {fid}')
        if training_example_content_sha256(example) != audit['training_example_content_sha256']:
            raise ValueError(f'Example content differs from audit: {fid}')
        if (audit['source_sha256']['yolo26x_weights'] != '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
                or audit['source_sha256']['split_manifest'] != batch['split_manifest_sha256']):
            raise ValueError(f'Frozen provenance differs: {fid}')
        for key in ('num_candidates', 'num_positive', 'num_negative', 'num_ignored'):
            if int(example[key]) != audit[key]:
                raise ValueError(f'Audit count differs: {fid} {key}')
        if len(frame['features']) != audit['num_feature_rows']:
            raise ValueError(f'Row count differs: {fid}')
        positives[fid] = int(example['num_positive'])
        input_hashes[fid] = dict(example=sha256(frame['example_path']), rows=sha256(frame['rows_path']), audit=sha256(audit_path))
        for key in total_keys:
            totals[key] += audit[key]
    if totals != summary['totals'] or not all(sum(positives[f] for f in group) > 0 for group in (train_ids, dev_ids)):
        raise ValueError('Totals differ or a partition lacks positives')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = dict(schema_version='clocs_128_internal_dev_split_v1', seed=args.seed,
                 train_frame_ids=train_ids, dev_frame_ids=dev_ids,
                 note='Earlier 16 pilot frames restricted to optimization; dev is for tuning, not final testing.')
    write_json(args.output_dir / 'split.json', split)
    write_json(args.output_dir / 'input_audit.json', dict(status='PASS', totals=totals, frames=input_hashes,
               batch_manifest_sha256=sha256(args.batch_manifest), summary_sha256=sha256(summary_path)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model = CLOCsFusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    protocol = dict(schema_version='clocs_128_training_protocol_v1',
        interpretation='Internal development candidate ranking; not official KITTI AP40.',
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sampling='All positives; half top PointPillars-score negatives, half random remaining negatives; ratio 0 uses all valid candidates.',
        loss='Focal alpha=0.25 gamma=2; mean over selected candidates per frame; evaluation uses all nonignored candidates.',
        reproducibility='Fixed seeds and deterministic selection; cross-device bitwise equality is not guaranteed.',
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                         gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None),
        source_sha256={str(p): sha256(p) for p in [Path(__file__), Path('fusion/clocs_sampling.py'), Path('fusion/clocs_network.py'), Path('fusion/clocs_training.py'), Path('scripts/train_clocs_pilot.py')]})
    write_json(args.output_dir / 'protocol.json', protocol)
    print('INPUTS PASS: 128 frames; 96 optimization / 32 internal development', flush=True)
    before = evaluate(model, root, dev_ids, device)
    write_json(args.output_dir / 'before_dev.json', before)
    history = []
    best_ap = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(train_ids)
        losses, selected_pos, selected_neg = [], 0, 0
        for fid in order:
            frame = load_frame(root, str(fid))
            labels = np.asarray(frame['example']['labels'])
            chosen = select_candidates(labels, frame['example']['candidate_scores_3d'], args.negative_ratio,
                                       args.seed, epoch, str(fid), args.background_cap)
            selected_pos += int((labels[chosen] == 1).sum())
            selected_neg += int((labels[chosen] == 0).sum())
            if len(chosen) == 0:
                raise ValueError(f'No valid training candidates: {fid}')
            optimizer.zero_grad(set_to_none=True)
            logits = predict(model, frame, device)
            chosen_tensor = torch.from_numpy(chosen).to(device)
            targets = torch.from_numpy(np.asarray(frame['example']['targets'], dtype=np.float32)).to(device)
            loss = sigmoid_focal_loss(logits[chosen_tensor], targets[chosen_tensor], reduction='mean')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            losses.append(float(loss.item()))
        dev = evaluate(model, root, dev_ids, device)
        record = dict(epoch=epoch, mean_training_selected_loss=float(np.mean(losses)),
                      selected_positive=selected_pos, selected_negative=selected_neg, dev=dev)
        history.append(record)
        print(json.dumps(record), flush=True)
        payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       epoch=epoch, split=split, protocol=protocol, dev=dev)
        temp = args.output_dir / 'last_checkpoint.tmp'
        torch.save(payload, temp)
        temp.replace(args.output_dir / 'last_checkpoint.pt')
        if dev['fusion_candidate_ap'] > best_ap:
            best_ap = dev['fusion_candidate_ap']
            torch.save(payload, args.output_dir / 'best_dev_checkpoint.pt')
        write_json(args.output_dir / 'history.json', history)
    checkpoint = args.output_dir / 'best_dev_checkpoint.pt'
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    reloaded = evaluate(model, root, dev_ids, device)
    error = abs(reloaded['fusion_candidate_ap'] - saved['dev']['fusion_candidate_ap'])
    if error > 1e-7:
        raise ValueError(f'Checkpoint reload mismatch: {error}')
    report = dict(status='PASS', interpretation=protocol['interpretation'], best_epoch=saved['epoch'],
                  before_dev=before, best_dev=reloaded, checkpoint_sha256=sha256(checkpoint),
                  reload_ap_absolute_error=error, split=split, protocol=protocol)
    write_json(args.output_dir / 'report.json', report)
    with (args.output_dir / 'learning_curve.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'selected_train_loss', 'full_dev_loss', 'pointpillars_candidate_ap', 'fusion_candidate_ap'])
        for r in history:
            writer.writerow([r['epoch'], r['mean_training_selected_loss'], r['dev']['mean_frame_focal_loss'], r['dev']['pointpillars_candidate_ap'], r['dev']['fusion_candidate_ap']])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
