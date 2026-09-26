#!/usr/bin/env python3
"""Independent CPU check of real CLOCs 3D IoUs with SciPy/Qhull.

The pinned MMDetection3D CUDA evaluator crashed natively in the available
Colab runtime. This independent geometry-engine diagnostic is explicitly
different from an official evaluator run or a KITTI AP result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.clocs_iou import best_camera_3d_iou
from fusion.clocs_training import build_candidate_targets
from fusion.kitti_labels import parse_kitti_label_file, prepare_clocs_car_ground_truth
from scripts.audit_clocs_iou_reference import (
    compare_iou, representative_indices, save_parity_figure,
)
from scripts.export_clocs_dense_train_frame import file_sha256


def qhull_footprint(box):
    """Return 2D halfspaces and bounds for a rectified camera box's x-z face."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    x, _, z, length, _, width, yaw = np.asarray(box, dtype=np.float64)
    local = np.array([[-length / 2, -width / 2],
                      [length / 2, -width / 2],
                      [length / 2, width / 2],
                      [-length / 2, width / 2]])
    rotation = Rotation.from_euler("z", -yaw).as_matrix()[:2, :2]
    corners = local @ rotation.T + (x, z)
    return ConvexHull(corners).equations, corners.min(axis=0), corners.max(axis=0)


def qhull_intersection_area(a, b):
    """Solve overlapping halfspaces and use Qhull's 2D area (hull.volume)."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    halfspaces = np.concatenate((a, b), axis=0)
    normals = halfspaces[:, :2]
    # Maximize the distance to every boundary to get a strict interior point.
    constraints = np.column_stack((normals, np.linalg.norm(normals, axis=1)))
    solution = linprog([0, 0, -1], A_ub=constraints,
                       b_ub=-halfspaces[:, 2],
                       bounds=[(None, None)] * 3, method="highs")
    if solution.status == 2:  # Disjoint halfspaces.
        return 0.0
    if not solution.success:
        raise RuntimeError(f"SciPy halfspace feasibility failed: {solution.message}")
    if solution.x[2] <= 1e-8:  # Boundary-only contact has zero area.
        return 0.0
    vertices = HalfspaceIntersection(halfspaces, solution.x[:2]).intersections
    return float(ConvexHull(vertices).volume)


def scipy_best_camera_3d_iou(candidates, gt_boxes):
    """Independent geometric IoU: Qhull halfspace area and camera y overlap."""
    candidates = np.asarray(candidates, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 7:
        raise ValueError("candidates must be [N, 7]")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
        raise ValueError("gt_boxes must be [M, 7]")
    if not np.isfinite(candidates).all() or not np.isfinite(gt_boxes).all():
        raise ValueError("3D boxes must contain finite values")
    if (candidates[:, 3:6] <= 0).any() or (gt_boxes[:, 3:6] <= 0).any():
        raise ValueError("3D box dimensions must be positive")
    result = np.zeros(candidates.shape[0], dtype=np.float64)
    gt_faces = [qhull_footprint(gt) for gt in gt_boxes]
    for i, candidate in enumerate(candidates):
        face, lower, upper = qhull_footprint(candidate)
        volume = float(np.prod(candidate[3:6]))
        for gt, (gt_face, gt_lower, gt_upper) in zip(gt_boxes, gt_faces):
            height = min(candidate[1], gt[1]) - max(
                candidate[1] - candidate[4], gt[1] - gt[4])
            if (height <= 0 or not np.all(upper > gt_lower)
                    or not np.all(gt_upper > lower)):
                continue
            area = qhull_intersection_area(face, gt_face)
            intersection = area * height
            if intersection > 0:
                result[i] = max(result[i], intersection /
                                (volume + np.prod(gt[3:6]) - intersection))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--audit", default="results/clocs_real_frame_000149_audit.json")
    parser.add_argument("--output", default="results/clocs_000149_iou_scipy_audit.json")
    parser.add_argument("--figure", default="results/clocs_000149_iou_scipy_parity.png")
    parser.add_argument("--per-group", type=int, default=64)
    args = parser.parse_args()

    import scipy
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if audit["frame_id"] != "000149" or audit["split"] != "train":
        raise ValueError("Only the frozen positive training frame is supported")
    root = Path(args.dataset)
    label = root / "label_2/000149.txt"
    calib = root / "calib/000149.txt"
    source = audit["source_sha256"]
    for path, expected in ((Path(args.candidates), source["dense_candidates"]),
                           (label, source["label"]), (calib, source["calibration"])):
        if file_sha256(path) != expected:
            raise ValueError(f"Input differs from the Mac audit: {path}")
    with np.load(args.candidates, allow_pickle=False) as archive:
        boxes = np.asarray(archive["boxes_camera"], dtype=np.float64)
        if (str(archive["frame_id"].item()) != "000149"
                or str(archive["checkpoint_sha256"].item()) != source["pointpillars_checkpoint"]
                or str(archive["config_sha256"].item()) != source["pointpillars_config"]
                or str(archive["split_manifest_sha256"].item()) != source["split_manifest"]):
            raise ValueError("Candidate export provenance differs from saved audit")
    if boxes.shape != (audit["num_candidates"], 7):
        raise ValueError("Candidate box shape differs from the Mac audit")

    records = parse_kitti_label_file(label)
    transform = build_kitti_lidar2cam(load_kitti_calibration(calib))
    gt = prepare_clocs_car_ground_truth(records, transform)["camera_boxes"]
    if gt.shape != (audit["car_gt_after_range_filter"], 7):
        raise ValueError("Retained Car GT count differs from the Mac audit")
    ours = best_camera_3d_iou(boxes, gt)
    counts = build_candidate_targets(ours)
    for key in ("num_positive", "num_negative", "num_ignored"):
        if counts[key] != audit[key]:
            raise ValueError(f"Full-frame {key} differs from the Mac audit")
    if (np.count_nonzero(ours) != audit["best_iou_nonzero"]
            or not np.isclose(float(ours.max()), audit["best_iou_max"], rtol=0, atol=1e-6)):
        raise ValueError("Full-frame IoU summary differs from the Mac audit")

    indices = representative_indices(ours, args.per_group)
    sampled = ours[indices]
    independent = scipy_best_camera_3d_iou(boxes[indices], gt)
    report = compare_iou(sampled, independent, indices)
    report.update({
        "schema_version": "clocs_scipy_geometry_parity_v1",
        "frame_id": "000149", "split": "train",
        "num_candidates": int(boxes.shape[0]), "num_car_gt": int(gt.shape[0]),
        "full_frame_counts_verified": True,
        "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "sample_strata": {
            "positive": int(np.count_nonzero(sampled >= .70)),
            "ignored": int(np.count_nonzero((sampled > .50) & (sampled < .70))),
            "overlapping_negative": int(np.count_nonzero((sampled > 0) & (sampled <= .50))),
            "zero_iou": int(np.count_nonzero(sampled == 0)),
        },
        "source_sha256": {"dense_candidates": source["dense_candidates"],
                          "label": source["label"], "calibration": source["calibration"],
                          "saved_audit": file_sha256(args.audit)},
        "reference": "Independent SciPy/Qhull 2D halfspace area plus camera y overlap",
        "scipy_version": scipy.__version__,
        "pinned_gpu_evaluator_status": "NOT RUN: pinned CUDA reference crashed natively (-11)",
        "interpretation": "Single real training-frame geometry check; not KITTI AP or trained fusion accuracy.",
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_parity_figure(Path(args.figure), sampled, independent,
                       report["status"], report["max_absolute_iou_error"],
                       reference_label="SciPy/Qhull reference 3D IoU")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved independent geometry figure: {args.figure}")
    if report["status"] != "PASS":
        raise SystemExit("Independent geometry comparison FAILED; do not train")


if __name__ == "__main__":
    main()
