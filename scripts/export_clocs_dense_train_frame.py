#!/usr/bin/env python3
"""Export one pinned, pre-NMS PointPillars frame on a CUDA runtime.

The result is a large NPZ for project storage, not a GitHub artifact.
Only canonical SECOND training frames are accepted.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import (
    extract_dense_pointpillars_candidates,
    load_pointpillars_model,
)
from scripts.validate_kitti_split import read_frame_ids, validate_split


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_training_frame(frame_id, manifest_path, train_path, val_path):
    if len(frame_id) != 6 or not frame_id.isdigit():
        raise ValueError("frame_id must be a six-digit KITTI identifier")
    split = validate_split(manifest_path, train_path, val_path)
    if not split["ok"]:
        raise ValueError(f"Frozen SECOND split mismatch: {split['mismatches']}")
    if frame_id not in set(read_frame_ids(train_path)):
        raise ValueError(f"{frame_id} is not a frozen SECOND training frame")
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="KITTI labelled training root")
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--config", required=True, help="Pinned MMDetection3D config")
    parser.add_argument("--checkpoint", required=True, help="Pinned PointPillars weights")
    parser.add_argument("--train", required=True, help="Canonical SECOND train.txt")
    parser.add_argument("--val", required=True, help="Canonical SECOND val.txt")
    parser.add_argument("--manifest", default="configs/splits/kitti_second_split.json")
    parser.add_argument("--evidence", default="results/pointpillars_evidence_manifest.json")
    parser.add_argument("--output", required=True, help="External-storage candidate NPZ")
    args = parser.parse_args()

    check_training_frame(args.frame_id, args.manifest, args.train, args.val)
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    config_path = evidence["model"]["config"]
    actual_config_path = Path(args.config).as_posix()
    if actual_config_path != config_path and not actual_config_path.endswith(
            "/" + config_path):
        raise ValueError(f"Expected the pinned PointPillars config: {config_path}")
    expected = evidence["model"]["checkpoint_sha256"]
    actual = file_sha256(args.checkpoint)
    if actual != expected:
        raise ValueError(f"PointPillars checkpoint SHA-256 mismatch: {actual}")

    root = Path(args.dataset)
    pointcloud = root / "velodyne" / f"{args.frame_id}.bin"
    calibration = root / "calib" / f"{args.frame_id}.txt"
    image = root / "image_2" / f"{args.frame_id}.png"
    for path in (pointcloud, calibration, image):
        if not path.is_file():
            raise FileNotFoundError(path)
    from PIL import Image
    with Image.open(image) as source:
        width, height = source.size

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This dense export requires an available CUDA GPU")
    model = load_pointpillars_model(args.config, args.checkpoint, device="cuda:0")
    candidates = extract_dense_pointpillars_candidates(
        model, pointcloud, calibration, width, height)
    fields = ("boxes_lidar", "boxes_camera", "scores_3d", "direction_labels",
              "projected_boxes", "distance_82", "candidate_index")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(args.config)
    split_sha256 = file_sha256(args.manifest)
    np.savez_compressed(output, frame_id=args.frame_id,
                        checkpoint_sha256=actual,
                        config_path=config_path,
                        config_sha256=config_sha256,
                        split_manifest_sha256=split_sha256,
                        **{key: candidates[key] for key in fields})
    print(json.dumps({
        "frame_id": args.frame_id, "split": "train",
        "num_candidates": candidates["num_candidates"],
        "feature_map_size": candidates["feature_map_size"],
        "checkpoint_sha256": actual,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_sha256,
        "candidate_npz_sha256": file_sha256(output),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
