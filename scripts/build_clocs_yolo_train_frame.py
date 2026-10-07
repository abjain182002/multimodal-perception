#!/usr/bin/env python3
"""Build one real CLOCs Car example from dense exports and pinned YOLO weights.

Outputs the auditable feature rows, a hash-verified training example and a
small JSON diagnostic. This single frame is a data-integrity gate, not AP.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.clocs import build_clocs_feature_rows, normalized_lidar_distance
from fusion.clocs_dataset import load_training_example_npz, save_training_example_npz
from fusion.kitti_labels import parse_kitti_label_file
from scripts.build_clocs_training_example import build_real_example, read_inputs
from scripts.export_clocs_dense_train_frame import check_training_frame, file_sha256


CAMERA_WEIGHT_SHA256 = {
    "yolov8x": "3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8",
    # User's local YOLO26x artifact, fingerprinted on 2026-09-28.
    # This records project provenance, not an upstream authenticity assertion.
    "yolo26x": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
}


def validate_camera_weights(camera_model, weights):
    actual = file_sha256(weights)
    if actual != CAMERA_WEIGHT_SHA256[camera_model]:
        raise ValueError(f"Weights do not match pinned {camera_model}: {actual}")
    return actual


def require_new_outputs(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def car_detections(result, names):
    boxes, scores = [], []
    for item in result.boxes:
        class_id = int(item.cls.item())
        if names[class_id] != "car":
            continue
        boxes.append(item.xyxy[0].detach().cpu().numpy().astype(np.float32))
        scores.append(float(item.conf.item()))
    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.stack(boxes), np.asarray(scores, dtype=np.float32)


def validate_export_provenance(candidates, evidence, split_manifest_sha256):
    expected = evidence["model"]["checkpoint_sha256"]
    if str(candidates["checkpoint_sha256"].item()) != expected:
        raise ValueError("Dense export does not use the pinned PointPillars checkpoint")
    if str(candidates["config_path"].item()) != evidence["model"]["config"]:
        raise ValueError("Dense export does not use the pinned PointPillars config")
    if str(candidates["split_manifest_sha256"].item()) != split_manifest_sha256:
        raise ValueError("Dense export split manifest differs from this repository")
    config_sha = str(candidates["config_sha256"].item())
    if len(config_sha) != 64 or any(c not in "0123456789abcdef" for c in config_sha):
        raise ValueError("Dense export config SHA-256 is invalid")
    return config_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="KITTI labelled training root")
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--candidates", required=True, help="Colab dense candidate NPZ")
    parser.add_argument("--yolo-weights", required=True, help="Frozen camera weights")
    parser.add_argument("--camera-model", choices=tuple(CAMERA_WEIGHT_SHA256),
                        default="yolov8x", help="Must match the pinned weight hash")
    parser.add_argument("--device", default="mps", help="Ultralytics inference device")
    parser.add_argument("--train", required=True, help="Canonical SECOND train.txt")
    parser.add_argument("--val", required=True, help="Canonical SECOND val.txt")
    parser.add_argument("--manifest", default="configs/splits/kitti_second_split.json")
    parser.add_argument("--evidence", default="results/kitti/pointpillars/pointpillars_evidence_manifest.json")
    parser.add_argument("--output", required=True, help="External-storage target NPZ")
    parser.add_argument("--rows", required=True, help="External-storage 2D/feature NPZ")
    parser.add_argument("--audit", required=True, help="Small JSON audit in results/")
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in
             (args.candidates, args.output, args.rows, args.audit)]
    if len(set(paths)) != len(paths):
        parser.error("--candidates, --output, --rows and --audit must be distinct")
    require_new_outputs(args.output, args.rows, args.audit)
    camera_sha = validate_camera_weights(args.camera_model, args.yolo_weights)

    check_training_frame(args.frame_id, args.manifest, args.train, args.val)
    candidates = read_inputs(args.candidates,
        ("boxes_lidar", "boxes_camera", "projected_boxes", "scores_3d",
         "distance_82", "candidate_index", "checkpoint_sha256",
         "config_path", "config_sha256", "split_manifest_sha256"), args.frame_id)
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    config_sha = validate_export_provenance(candidates, evidence,
                                            file_sha256(args.manifest))
    np.testing.assert_allclose(candidates["distance_82"],
        normalized_lidar_distance(candidates["boxes_lidar"]), rtol=1e-5, atol=1e-6,
        err_msg="Dense LiDAR boxes do not match their distance features")

    root = Path(args.dataset)
    image = root / "image_2" / f"{args.frame_id}.png"
    calibration = root / "calib" / f"{args.frame_id}.txt"
    labels = root / "label_2" / f"{args.frame_id}.txt"
    weights = Path(args.yolo_weights)
    for path in (image, calibration, labels, weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    import ultralytics
    from ultralytics import YOLO
    detector = YOLO(str(weights))
    prediction = detector.predict(source=str(image), device=args.device,
                                  conf=0.25, verbose=False)[0]
    boxes_2d, scores_2d = car_detections(prediction, detector.names)
    rows = build_clocs_feature_rows(candidates["projected_boxes"],
                                    candidates["scores_3d"], boxes_2d,
                                    scores_2d, candidates["distance_82"])
    raw_rows = {"boxes_2d": boxes_2d, "scores_2d": scores_2d,
                "features": rows["features"],
                "candidate_index": rows["candidate_index"],
                "detector_2d_index": rows["detector_2d_index"],
                "row_type": rows["row_type"]}

    records = parse_kitti_label_file(labels)
    transform = build_kitti_lidar2cam(load_kitti_calibration(calibration))
    example, gt = build_real_example(args.frame_id, "train", candidates,
                                     raw_rows, records, transform)
    output = Path(args.output)
    rows_path = Path(args.rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rows_path, frame_id=args.frame_id,
                        camera_model=args.camera_model,
                        camera_weights_sha256=camera_sha, **raw_rows)
    saved = save_training_example_npz(example, output)
    load_training_example_npz(output, verify_integrity=True)

    summary = {
        "schema_version": ("clocs_real_train_frame_audit_v1"
                           if args.camera_model == "yolov8x"
                           else "clocs_real_train_frame_audit_v2"),
        "frame_id": args.frame_id, "split": "train",
        "camera_model": args.camera_model,
        "ultralytics_version": ultralytics.__version__,
        "camera_device": args.device,
        "camera_confidence_threshold": 0.25,
        "car_gt_after_range_filter": gt["num_car_after_range_filter"],
        f"{args.camera_model}_car_detections": int(boxes_2d.shape[0]),
        "num_candidates": example["num_candidates"],
        "num_feature_rows": example["num_rows"],
        "num_positive": example["num_positive"],
        "num_negative": example["num_negative"],
        "num_ignored": example["num_ignored"],
        "best_iou_nonzero": int(np.count_nonzero(example["best_iou"])),
        "best_iou_max": float(example["best_iou"].max()),
        "source_sha256": {
            "dense_candidates": file_sha256(args.candidates),
            "pointpillars_checkpoint": str(candidates["checkpoint_sha256"].item()),
            "pointpillars_config": config_sha,
            f"{args.camera_model}_weights": camera_sha,
            "image": file_sha256(image),
            "calibration": file_sha256(calibration),
            "label": file_sha256(labels),
            "split_manifest": file_sha256(args.manifest),
            "feature_rows": file_sha256(rows_path),
        },
        "training_example_content_sha256": saved["content_sha256"],
        "interpretation": "One training-frame integrity check; not KITTI AP or a trained fusion result.",
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved audit: {audit_path}")
    print(f"Saved large examples outside Git: {output}, {rows_path}")


if __name__ == "__main__":
    main()

