#!/usr/bin/env python3
"""Build one genuine KITTI Car CLOCs example from aligned dense boxes.

Both input NPZ files must identify the same frame. The candidate archive needs
the adapter's boxes_camera, scores_3d, distance_82, and candidate_index. The
feature archive needs features and candidate_index from the same extraction.
Saved structural smoke archives lack 3D boxes and cannot pass this gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.clocs_dataset import (
    build_training_example,
    load_training_example_npz,
    save_training_example_npz,
)
from fusion.clocs_iou import best_camera_3d_iou
from fusion.kitti_labels import parse_kitti_label_file, prepare_clocs_car_ground_truth
from scripts.validate_kitti_split import read_frame_ids, validate_split


def read_inputs(path, fields, frame_id):
    with np.load(path, allow_pickle=False) as archive:
        missing = (set(fields) | {"frame_id"}) - set(archive.files)
        if missing:
            raise ValueError(f"{path} missing required fields: {sorted(missing)}")
        value = archive["frame_id"]
        if value.shape != () or str(value.item()) != frame_id:
            raise ValueError(f"{path} frame_id does not match {frame_id}")
        return {key: np.array(archive[key], copy=True) for key in fields}


def build_real_example(frame_id, split, candidates, feature_archive, records,
                       lidar_to_camera):
    boxes = np.asarray(candidates["boxes_camera"])
    scores = np.asarray(candidates["scores_3d"])
    distances = np.asarray(candidates["distance_82"])
    dense_index = np.asarray(candidates["candidate_index"])
    features = np.asarray(feature_archive["features"])
    row_index = np.asarray(feature_archive["candidate_index"])

    if boxes.ndim != 2 or boxes.shape[1] != 7 or boxes.shape[0] == 0:
        raise ValueError("boxes_camera must have shape [N, 7] with N > 0")
    num_candidates = boxes.shape[0]
    if scores.shape != (num_candidates,) or distances.shape != (num_candidates,):
        raise ValueError("scores_3d and distance_82 must align with boxes_camera")
    if dense_index.shape != (num_candidates,) or not np.issubdtype(dense_index.dtype, np.integer):
        raise ValueError("candidate_index must be an integer index for every dense box")
    if not np.array_equal(dense_index, np.arange(num_candidates)):
        raise ValueError("candidate_index must preserve original dense box order")
    if features.ndim != 2 or features.shape[1] != 4 or features.shape[0] == 0:
        raise ValueError("features must have shape [R, 4] with R > 0")
    if row_index.shape != (features.shape[0],) or not np.issubdtype(row_index.dtype, np.integer):
        raise ValueError("feature candidate_index must have one integer per row")
    if not np.isfinite(features).all() or not np.isfinite(scores).all() or not np.isfinite(distances).all():
        raise ValueError("features, scores_3d, and distance_82 must be finite")
    if np.any(row_index < 0) or np.any(row_index >= num_candidates):
        raise ValueError("feature candidate_index is outside dense candidate range")
    if not np.allclose(features[:, 1], scores[row_index], rtol=0, atol=1e-6):
        raise ValueError("feature rows do not match candidate 3D scores")
    if not np.allclose(features[:, 3], distances[row_index], rtol=0, atol=1e-6):
        raise ValueError("feature rows do not match candidate distances")

    ground_truth = prepare_clocs_car_ground_truth(records, lidar_to_camera)
    best_iou = best_camera_3d_iou(boxes, ground_truth["camera_boxes"])
    example = build_training_example(frame_id, split, features, row_index,
                                     scores, best_iou)
    return example, ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="KITTI labelled training root")
    parser.add_argument("--frame-id", required=True, help="Six-digit KITTI frame ID")
    parser.add_argument("--candidates", required=True, help="Dense PointPillars NPZ")
    parser.add_argument("--features", required=True, help="CLOCs feature rows NPZ")
    parser.add_argument("--train", required=True, help="Frozen SECOND train.txt")
    parser.add_argument("--val", required=True, help="Frozen SECOND val.txt")
    parser.add_argument("--manifest", default="configs/splits/kitti_second_split.json")
    parser.add_argument("--output", required=True, help="Per-frame training example NPZ")
    args = parser.parse_args()
    if len(args.frame_id) != 6 or not args.frame_id.isdigit():
        parser.error("--frame-id must be a six-digit KITTI ID")
    split = validate_split(args.manifest, args.train, args.val)
    if not split["ok"]:
        raise ValueError(f"Frozen split mismatch: {split['mismatches']}")
    train_ids = set(read_frame_ids(args.train))
    val_ids = set(read_frame_ids(args.val))
    if args.frame_id not in train_ids | val_ids:
        raise ValueError("frame_id is not in the frozen KITTI train/validation split")
    split_name = "train" if args.frame_id in train_ids else "validation"
    candidates = read_inputs(args.candidates,
                             ("boxes_camera", "scores_3d", "distance_82", "candidate_index"),
                             args.frame_id)
    features = read_inputs(args.features, ("features", "candidate_index"),
                           args.frame_id)
    dataset = Path(args.dataset)
    records = parse_kitti_label_file(dataset / "label_2" / (args.frame_id + ".txt"))
    calibration = load_kitti_calibration(dataset / "calib" / (args.frame_id + ".txt"))
    example, gt = build_real_example(args.frame_id, split_name, candidates, features,
                                     records, build_kitti_lidar2cam(calibration))
    saved = save_training_example_npz(example, args.output)
    load_training_example_npz(args.output, verify_integrity=True)
    print(json.dumps({"frame_id": args.frame_id, "split": split_name,
                      "num_gt_after_range_filter": gt["num_car_after_range_filter"],
                      "num_candidates": example["num_candidates"],
                      "num_positive": example["num_positive"],
                      "num_negative": example["num_negative"],
                      "num_ignored": example["num_ignored"],
                      "content_sha256": saved["content_sha256"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
