#!/usr/bin/env python3
"""Train and audit a small, reproducible CLOCs fusion pilot.

The PointPillars and YOLO26x outputs are frozen.  Only the CLOCs fusion
network is optimized.  This script is deliberately a pilot experiment: the
16 selected KITTI training frames are split deterministically into an
optimization subset and a held-out subset, and the report labels that scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from fusion.clocs_dataset import load_training_example_npz
from fusion.clocs_network import CLOCsFusionNet
from fusion.clocs_training import build_classification_weights, sigmoid_focal_loss


SCHEMA = "clocs_yolo26_pilot_training_v1"
DEFAULT_SEED = 20260928


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected scalar NPZ field")
    return array.item()


def load_frame(data_dir: Path, frame_id: str) -> dict:
    example_path = data_dir / "examples" / f"{frame_id}_real_targets.npz"
    rows_path = data_dir / "rows" / f"{frame_id}_clocs_rows.npz"
    if not example_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"missing example or rows archive for {frame_id}")

    example = load_training_example_npz(example_path, verify_integrity=True)
    with np.load(rows_path, allow_pickle=False) as archive:
        names = set(archive.files)
        feature_key = "features" if "features" in names else "feature_rows"
        index_key = "candidate_index" if "candidate_index" in names else "candidate_indices"
        if feature_key not in names or index_key not in names:
            raise ValueError(f"rows archive for {frame_id} has no feature/index fields")
        features = np.array(archive[feature_key], dtype=np.float32, copy=True)
        candidate_index = np.array(archive[index_key], dtype=np.int64, copy=True)
        camera_model = str(scalar(archive["camera_model"])) if "camera_model" in names else "unknown"

    if features.shape != example["feature_rows"].shape or not np.array_equal(features, example["feature_rows"]):
        raise ValueError(f"feature rows do not exactly match canonical example for {frame_id}")
    if candidate_index.shape != example["candidate_indices"].shape or not np.array_equal(candidate_index, example["candidate_indices"]):
        raise ValueError(f"candidate indices do not exactly match canonical example for {frame_id}")
    return {
        "frame_id": frame_id,
        "example_path": example_path,
        "rows_path": rows_path,
        "example": example,
        "features": features,
        "candidate_index": candidate_index,
        "camera_model": camera_model,
    }


def frame_ids(data_dir: Path) -> list[str]:
    ids = sorted(path.name.split("_", 1)[0] for path in (data_dir / "examples").glob("*_real_targets.npz"))
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError(f"expected 16 unique packaged examples, found {len(ids)}")
    return ids


def make_split(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(ids))
    for holdout_count in (4, 5, 3):
        holdout = order[:holdout_count]
        train = order[holdout_count:]
        if train and holdout:
            return sorted(train), sorted(holdout)
    raise RuntimeError("could not construct a non-empty split")


def candidate_scores(frame: dict) -> np.ndarray:
    values = np.asarray(frame["example"]["candidate_scores_3d"], dtype=np.float32)
    if not np.isfinite(values).all() or values.ndim != 1:
        raise ValueError("invalid candidate scores")
    return values


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum((cumulative / ranks) * ordered) / positives)


def evaluate(model: CLOCsFusionNet, frames: list[dict], device: torch.device) -> dict:
    model.eval()
    all_labels, all_baseline, all_fused = [], [], []
    losses = []
    with torch.no_grad():
        for frame in frames:
            example = frame["example"]
            features = torch.from_numpy(frame["features"]).to(device=device)
            indices = torch.from_numpy(frame["candidate_index"]).to(device=device)
            labels_np = np.asarray(example["labels"], dtype=np.int8)
            targets = torch.from_numpy(np.asarray(example["targets"], dtype=np.float32)).to(device=device)
            labels = torch.from_numpy(labels_np.astype(np.int64)).to(device=device)
            weights = build_classification_weights(labels)["weights"].to(device)
            output = model(features, indices, int(example["num_candidates"]))
            per_item = sigmoid_focal_loss(output["candidate_logits"], targets, weights=weights, reduction="none")
            losses.append(float(per_item.sum().item() / max(float(weights.sum().item()), 1.0)))
            valid = labels_np >= 0
            all_labels.append(labels_np[valid])
            all_baseline.append(candidate_scores(frame)[valid])
            all_fused.append(torch.sigmoid(output["candidate_logits"]).cpu().numpy()[valid])
    labels = np.concatenate(all_labels)
    baseline = np.concatenate(all_baseline)
    fused = np.concatenate(all_fused)
    return {
        "frames": len(frames),
        "candidates": int(labels.size),
        "positive_candidates": int(labels.sum()),
        "mean_normalized_focal_loss": float(np.mean(losses)),
        "baseline_pointpillars_ap": average_precision(labels, baseline),
        "fused_clocs_ap": average_precision(labels, fused),
    }


def train_epoch(model, frames, optimizer, device):
    model.train()
    losses = []
    gradients = []
    for frame in frames:
        example = frame["example"]
        features = torch.from_numpy(frame["features"]).to(device=device)
        indices = torch.from_numpy(frame["candidate_index"]).to(device=device)
        labels = torch.from_numpy(np.asarray(example["labels"], dtype=np.int64)).to(device=device)
        targets = torch.from_numpy(np.asarray(example["targets"], dtype=np.float32)).to(device=device)
        weights = build_classification_weights(labels)["weights"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features, indices, int(example["num_candidates"]))
        per_item = sigmoid_focal_loss(output["candidate_logits"], targets, weights=weights, reduction="none")
        loss = per_item.sum() / torch.clamp(weights.sum(), min=1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss on frame {frame['frame_id']}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item())
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient on frame {frame['frame_id']}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradients.append(grad_norm)
    return {"mean_normalized_focal_loss": float(np.mean(losses)), "max_gradient_norm": float(max(gradients))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Extracted Kaggle input package directory")
    parser.add_argument("--output-dir", required=True, help="Writable output directory")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto (default: cuda when available)")
    args = parser.parse_args()
    if args.epochs <= 0 or args.epochs > 100:
        parser.error("--epochs must be between 1 and 100")
    data_dir, output_dir = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    ids = frame_ids(data_dir)
    train_ids, holdout_ids = make_split(ids, args.seed)
    frames = {frame_id: load_frame(data_dir, frame_id) for frame_id in ids}
    train_frames = [frames[frame_id] for frame_id in train_ids]
    holdout_frames = [frames[frame_id] for frame_id in holdout_ids]
    if sum(int(frames[f]["example"]["num_positive"]) for f in holdout_ids) == 0:
        raise ValueError("held-out pilot subset contains no positive candidates")

    model = CLOCsFusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    before = {"train": evaluate(model, train_frames, device), "holdout": evaluate(model, holdout_frames, device)}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_step = train_epoch(model, train_frames, optimizer, device)
        evaluation = {"train": evaluate(model, train_frames, device), "holdout": evaluate(model, holdout_frames, device)}
        record = {"epoch": epoch, "train_step": train_step, "evaluation": evaluation}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    after = {"train": evaluate(model, train_frames, device), "holdout": evaluate(model, holdout_frames, device)}

    checkpoint_path = output_dir / "clocs_yolo26_pilot_checkpoint.pt"
    torch.save({"schema_version": SCHEMA, "seed": args.seed, "train_frame_ids": train_ids,
                "holdout_frame_ids": holdout_ids, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "epochs": args.epochs,
                "learning_rate": args.learning_rate}, checkpoint_path)
    with checkpoint_path.open("rb") as handle:
        checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()

    report = {"schema_version": SCHEMA, "status": "PASS", "interpretation": "16-frame KITTI training pilot; not full KITTI AP and not a held-out official benchmark.",
              "camera_model": "yolo26x", "device": str(device), "seed": args.seed, "epochs": args.epochs,
              "learning_rate": args.learning_rate, "all_frame_ids": ids, "train_frame_ids": train_ids,
              "holdout_frame_ids": holdout_ids, "train_positive_candidates": int(sum(frames[f]["example"]["num_positive"] for f in train_ids)),
              "holdout_positive_candidates": int(sum(frames[f]["example"]["num_positive"] for f in holdout_ids)),
              "before": before, "after": after, "history": history,
              "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
              "input_sha256": {"summary": sha256(data_dir / "summary.json"), "manifest": sha256(data_dir / "manifest.json"),
                               "examples": {f: sha256(frames[f]["example_path"]) for f in ids},
                               "rows": {f: sha256(frames[f]["rows_path"]) for f in ids}}}
    report_path = output_dir / "clocs_yolo26_pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    curve_path = output_dir / "clocs_yolo26_pilot_learning_curve.csv"
    with curve_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,train_ap,holdout_loss,holdout_ap\n")
        for record in history:
            handle.write(f"{record['epoch']},{record['train_step']['mean_normalized_focal_loss']},{record['evaluation']['train']['fused_clocs_ap']},{record['evaluation']['holdout']['mean_normalized_focal_loss']},{record['evaluation']['holdout']['fused_clocs_ap']}\n")
    print(json.dumps({"status": "PASS", "device": str(device), "checkpoint": str(checkpoint_path), "report": str(report_path), "curve": str(curve_path), "before_holdout": before["holdout"], "after_holdout": after["holdout"]}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
