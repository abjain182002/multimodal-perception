#!/usr/bin/env python3
"""Compare real CLOCs targets with the pinned MMDetection3D KITTI 3D IoU.

Run in the pinned CUDA environment. This checks a deterministic sample of
genuine train-frame anchors against a separate, GPU-based overlap kernel;
it is not an object-detection evaluation or a KITTI AP result.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.clocs_iou import best_camera_3d_iou
from fusion.clocs_training import build_candidate_targets
from fusion.kitti_labels import parse_kitti_label_file, prepare_clocs_car_ground_truth
from scripts.export_clocs_dense_train_frame import file_sha256


PINNED_MMDET3D_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"


def representative_indices(best_iou, per_group=64):
    """Cover every target class, nonoverlapping boxes and threshold edges."""
    values = np.asarray(best_iou, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("best_iou must be a finite one-dimensional array")
    if not 1 <= per_group <= 256:
        raise ValueError("per_group must be in [1, 256]")
    groups = (
        np.flatnonzero(values >= 0.70),
        np.flatnonzero((values > 0.50) & (values < 0.70)),
        np.flatnonzero((values > 0) & (values <= 0.50)),
        np.flatnonzero(values == 0),
    )
    selected = []
    for group in groups:
        if group.size:
            positions = np.linspace(0, group.size - 1,
                                    min(per_group, group.size), dtype=int)
            selected.extend(group[positions].tolist())
    if values.size:
        selected.extend([int(np.argmax(values)),
                         int(np.argmin(np.abs(values - 0.50))),
                         int(np.argmin(np.abs(values - 0.70)))])
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def compare_iou(ours, reference, indices, atol=5e-4):
    ours = np.asarray(ours, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if ours.shape != reference.shape or ours.shape != indices.shape:
        raise ValueError("Sample, reference and indices must be aligned")
    if not np.isfinite(reference).all() or (reference < 0).any() or (reference > 1.0001).any():
        raise ValueError("Reference IoU contains invalid values")
    errors = np.abs(ours - reference)
    a = build_candidate_targets(ours)["labels"]
    b = build_candidate_targets(reference)["labels"]
    # A float32/float64 threshold-edge difference is acceptable only inside
    # the numerical tolerance. Report it rather than hiding it.
    disagreements = np.flatnonzero(a != b)
    failures = np.flatnonzero(errors > atol)
    examples = np.argsort(-errors, kind="stable")[:10]
    return {
        "status": "PASS" if not failures.size and not disagreements.size else "FAIL",
        "sample_size": int(indices.size),
        "absolute_tolerance": atol,
        "max_absolute_iou_error": float(errors.max()) if errors.size else 0.0,
        "mean_absolute_iou_error": float(errors.mean()) if errors.size else 0.0,
        "out_of_tolerance": int(failures.size),
        "label_disagreements": int(disagreements.size),
        "worst_pairs": [{"candidate_index": int(indices[i]),
                         "our_iou": float(ours[i]),
                         "reference_iou": float(reference[i]),
                         "absolute_error": float(errors[i])} for i in examples],
    }


def save_parity_figure(path, ours, reference, status, max_error):
    """Render measured parity; each point is one genuine train-frame anchor."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ours = np.asarray(ours)
    reference = np.asarray(reference)
    fig, ax = plt.subplots(figsize=(6.2, 5.5), layout="constrained")
    ax.plot([0, 1], [0, 1], color="0.3", linewidth=1, label="Exact agreement")
    for name, mask, color in (
        ("Negative or zero", ours <= .50, "#377eb8"),
        ("Ignored", (ours > .50) & (ours < .70), "#e69f00"),
        ("Positive", ours >= .70, "#009e73"),
    ):
        if mask.any():
            ax.scatter(ours[mask], reference[mask], s=24, alpha=.7,
                       color=color, label=f"{name} (n={mask.sum()})")
    for threshold in (.50, .70):
        ax.axvline(threshold, linewidth=.8, linestyle=":", color="0.55")
        ax.axhline(threshold, linewidth=.8, linestyle=":", color="0.55")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02),
           xlabel="Our camera-space 3D IoU", ylabel="Pinned KITTI evaluator 3D IoU",
           title=f"KITTI train 000149 | {len(ours)} sampled anchors | {status}")
    ax.text(.03, .97, f"Maximum absolute difference: {max_error:.6g}",
            ha="left", va="top", transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": .9})
    ax.legend(loc="lower right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="KITTI training directory containing calib/ and label_2/")
    parser.add_argument("--candidates", required=True, help="Original dense export NPZ")
    parser.add_argument("--audit", default="results/clocs_real_frame_000149_audit.json")
    parser.add_argument("--mmdet3d-root", default="/content/mmdetection3d")
    parser.add_argument("--output", default="results/clocs_000149_iou_reference_audit.json")
    parser.add_argument("--figure", default="results/clocs_000149_iou_parity.png")
    parser.add_argument("--per-group", type=int, default=64)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    frame = audit["frame_id"]
    if audit["split"] != "train" or frame != "000149":
        raise ValueError("This audit must use the frozen positive train frame 000149")
    data = Path(args.dataset)
    label = data / "label_2" / f"{frame}.txt"
    calib = data / "calib" / f"{frame}.txt"
    source = audit["source_sha256"]
    for path, expected in ((Path(args.candidates), source["dense_candidates"]),
                           (label, source["label"]), (calib, source["calibration"])):
        if file_sha256(path) != expected:
            raise ValueError(f"SHA-256 does not match saved frame audit: {path}")

    with np.load(args.candidates, allow_pickle=False) as saved:
        if str(saved["frame_id"].item()) != frame:
            raise ValueError("Candidate export frame mismatch")
        if str(saved["checkpoint_sha256"].item()) != source["pointpillars_checkpoint"]:
            raise ValueError("PointPillars checkpoint mismatch")
        if str(saved["config_sha256"].item()) != source["pointpillars_config"]:
            raise ValueError("PointPillars config mismatch")
        if str(saved["split_manifest_sha256"].item()) != source["split_manifest"]:
            raise ValueError("Train split manifest mismatch")
        boxes = np.array(saved["boxes_camera"], dtype=np.float64)
    if boxes.shape != (audit["num_candidates"], 7):
        raise ValueError("Dense candidate count/shape mismatch")

    records = parse_kitti_label_file(label)
    transform = build_kitti_lidar2cam(load_kitti_calibration(calib))
    gt = prepare_clocs_car_ground_truth(records, transform)
    gt_boxes = np.asarray(gt["camera_boxes"], dtype=np.float64)
    if gt_boxes.shape != (audit["car_gt_after_range_filter"], 7) or not gt_boxes.size:
        raise ValueError("Retained Car ground truth differs from the saved audit")

    ours = best_camera_3d_iou(boxes, gt_boxes)
    targets = build_candidate_targets(ours)
    for key, expected in (("num_positive", audit["num_positive"]),
                          ("num_negative", audit["num_negative"]),
                          ("num_ignored", audit["num_ignored"])):
        if targets[key] != expected:
            raise ValueError(f"Full-frame recomputation differs in {key}")
    if np.count_nonzero(ours) != audit["best_iou_nonzero"] or not np.isclose(
            float(ours.max()), audit["best_iou_max"], rtol=0, atol=1e-6):
        raise ValueError("Full-frame IoU recomputation differs from the Mac audit")

    root = Path(args.mmdet3d_root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != PINNED_MMDET3D_COMMIT:
        raise ValueError(f"MMDetection3D checkout must be {PINNED_MMDET3D_COMMIT}, got {commit}")
    sys.path.insert(0, str(root))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned KITTI reference evaluator requires CUDA")
    from mmdet3d.evaluation.functional.kitti_utils.eval import d3_box_overlap
    kernel = root / "mmdet3d/evaluation/functional/kitti_utils/eval.py"

    indices = representative_indices(ours, args.per_group)
    reference = d3_box_overlap(boxes[indices], gt_boxes).max(axis=1)
    report = compare_iou(ours[indices], reference, indices)
    report.update({
        "schema_version": "clocs_real_iou_reference_audit_v1",
        "frame_id": frame,
        "split": "train",
        "num_candidates": int(boxes.shape[0]),
        "num_car_gt": int(gt_boxes.shape[0]),
        "full_frame_counts_verified": True,
        "sampling": "up to 64 evenly spaced anchors in each positive/ignored/overlapping-negative/zero-IoU group, plus max and threshold-nearest anchors",
        "reference": "Pinned MMDetection3D KITTI d3_box_overlap (GPU), camera-coordinate rotated 3D IoU",
        "mmdetection3d_commit": commit,
        "source_sha256": {"candidates": source["dense_candidates"],
                          "label": source["label"],
                          "calibration": source["calibration"],
                          "saved_audit": file_sha256(args.audit),
                          "reference_evaluator": file_sha256(kernel)},
        "interpretation": "One real train-frame geometry parity check; not KITTI AP or fusion accuracy.",
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_parity_figure(Path(args.figure), ours[indices], reference,
                       report["status"], report["max_absolute_iou_error"])
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved parity figure: {args.figure}")
    if report["status"] != "PASS":
        raise SystemExit("Reference parity FAILED; inspect the saved discrepancies before training")


if __name__ == "__main__":
    main()
