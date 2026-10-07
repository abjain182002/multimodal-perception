#!/usr/bin/env python3
"""Audit CLOCs Car ground truth for the frozen KITTI train/val split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.pointpillars_adapter import (
    build_kitti_lidar2cam,
    load_kitti_calibration,
)
from fusion.kitti_labels import (
    parse_kitti_label_file,
    prepare_clocs_car_ground_truth,
)
from scripts.validate_kitti_split import (
    read_frame_ids,
    validate_split,
)


def audit_split(dataset_dir, frame_ids):
    """Count raw and range-filtered Car GT for every listed frame."""
    dataset_dir = Path(dataset_dir)
    stats = {
        "frames": len(frame_ids),
        "car_before_range_filter": 0,
        "car_after_range_filter": 0,
        "car_removed_by_range_filter": 0,
        "frames_without_car_before_filter": 0,
        "frames_without_car_after_filter": 0,
        "frames_with_removed_car": 0,
        "max_car_after_filter_in_one_frame": 0,
    }

    for frame_id in frame_ids:
        label_path = dataset_dir / "label_2" / f"{frame_id}.txt"
        calib_path = dataset_dir / "calib" / f"{frame_id}.txt"
        try:
            records = parse_kitti_label_file(label_path)
            calibration = load_kitti_calibration(calib_path)
            gt = prepare_clocs_car_ground_truth(
                records,
                build_kitti_lidar2cam(calibration),
            )
        except (ValueError, FileNotFoundError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Frame {frame_id}: {exc}") from exc

        before = gt["num_car_before_range_filter"]
        after = gt["num_car_after_range_filter"]
        if (
            before != sum(record["name"] == "Car" for record in records)
            or after > before
            or gt["camera_boxes"].shape != (after, 7)
            or gt["lidar_boxes"].shape != (after, 7)
            or not np.isfinite(gt["camera_boxes"]).all()
            or not np.isfinite(gt["lidar_boxes"]).all()
        ):
            raise ValueError(f"Frame {frame_id}: inconsistent Car GT")

        stats["car_before_range_filter"] += before
        stats["car_after_range_filter"] += after
        stats["frames_without_car_before_filter"] += int(before == 0)
        stats["frames_without_car_after_filter"] += int(after == 0)
        stats["frames_with_removed_car"] += int(after < before)
        stats["max_car_after_filter_in_one_frame"] = max(
            stats["max_car_after_filter_in_one_frame"], after
        )

    stats["car_removed_by_range_filter"] = (
        stats["car_before_range_filter"] - stats["car_after_range_filter"]
    )
    return stats


def audit_dataset(dataset_dir, manifest_path, train_path, val_path):
    """Validate canonical splits before reading any frame's ground truth."""
    split = validate_split(manifest_path, train_path, val_path)
    if not split["ok"]:
        raise ValueError(f"Split manifest mismatch: {split['mismatches']}")

    train_ids = read_frame_ids(train_path)
    val_ids = read_frame_ids(val_path)
    result = {
        "schema_version": "kitti_clocs_gt_audit_v1",
        "dataset": "KITTI Object Detection labelled training directory",
        "split_manifest_sha256": hashlib.sha256(
            Path(manifest_path).read_bytes()
        ).hexdigest(),
        "split": split["observed"],
        "train": audit_split(dataset_dir, train_ids),
        "validation": audit_split(dataset_dir, val_ids),
    }
    result["total"] = {
        key: result["train"][key] + result["validation"][key]
        for key in result["train"]
        if key != "max_car_after_filter_in_one_frame"
    }
    result["total"]["max_car_after_filter_in_one_frame"] = max(
        result["train"]["max_car_after_filter_in_one_frame"],
        result["validation"]["max_car_after_filter_in_one_frame"],
    )
    if result["total"]["frames"] != split["observed"]["union_count"]:
        raise ValueError("Audited frame count does not match split union")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="KITTI labelled training directory containing calib/ and label_2/",
    )
    parser.add_argument("--train", required=True, help="Frozen train.txt")
    parser.add_argument("--val", required=True, help="Frozen val.txt")
    parser.add_argument(
        "--manifest", default="configs/splits/kitti_second_split.json",
    )
    parser.add_argument(
        "--output", default="results/kitti/clocs/training_exports/kitti_clocs_gt_audit.json",
    )
    args = parser.parse_args()

    result = audit_dataset(
        args.dataset, args.manifest, args.train, args.val
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"train": result["train"], "validation": result["validation"],
         "total": result["total"]},
        indent=2,
        sort_keys=True,
    ))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

