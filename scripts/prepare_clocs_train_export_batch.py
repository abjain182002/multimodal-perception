#!/usr/bin/env python3
"""Select a reproducible train-only KITTI batch for dense CLOCs GPU export.

This is an input audit and export queue, not a model-training or AP result.
KITTI validation frames are never eligible. The deliberately balanced pilot
batch is not representative of the complete training distribution.
"""

import argparse
import hashlib
import json
from pathlib import Path

from adapters.pointpillars_adapter import build_kitti_lidar2cam, load_kitti_calibration
from fusion.kitti_labels import parse_kitti_label_file, prepare_clocs_car_ground_truth
from scripts.export_clocs_dense_train_frame import file_sha256
from scripts.validate_kitti_split import read_frame_ids, validate_split


GROUPS = ("zero", "one_two", "three_six", "seven_plus")


def car_group(count):
    if count == 0:
        return "zero"
    if count <= 2:
        return "one_two"
    if count <= 6:
        return "three_six"
    return "seven_plus"


def build_export_batch(dataset, train, val, manifest, evidence, size=16,
                       seed=20260926, include=()):
    """Audit the frozen split and select a bounded, balanced training pilot."""
    dataset = Path(dataset)
    train = Path(train)
    val = Path(val)
    manifest = Path(manifest)
    evidence = Path(evidence)
    if size < 1:
        raise ValueError("size must be positive")
    split = validate_split(manifest, train, val)
    if not split["ok"]:
        raise ValueError(f"Frozen SECOND split mismatch: {split['mismatches']}")
    train_ids = read_frame_ids(train)
    val_ids = set(read_frame_ids(val))
    anchors = list(include)
    if len(set(anchors)) != len(anchors):
        raise ValueError("Duplicate included frame")
    if len(anchors) > size or size > len(train_ids):
        raise ValueError("size cannot fit included frames or exceeds train split")
    if not set(anchors).issubset(train_ids) or set(anchors) & val_ids:
        raise ValueError("Included frame is not in the frozen training split")

    frame_gt = {}
    pools = {name: [] for name in GROUPS}
    for frame_id in train_ids:
        label = dataset / "label_2" / f"{frame_id}.txt"
        calib = dataset / "calib" / f"{frame_id}.txt"
        records = parse_kitti_label_file(label)
        transform = build_kitti_lidar2cam(load_kitti_calibration(calib))
        gt = prepare_clocs_car_ground_truth(records, transform)
        before = gt["num_car_before_range_filter"]
        after = gt["num_car_after_range_filter"]
        if after > before:
            raise ValueError(f"GT range filtering inconsistent: {frame_id}")
        frame_gt[frame_id] = (before, after)
        pools[car_group(after)].append(frame_id)

    selected = list(anchors)
    already = set(anchors)
    for name in GROUPS:
        pools[name].sort(key=lambda frame: hashlib.sha256(
            f"{seed}:{frame}".encode("ascii")).hexdigest())
    positions = {name: 0 for name in GROUPS}
    while len(selected) < size:
        progressed = False
        for name in GROUPS:
            group = pools[name]
            while positions[name] < len(group) and group[positions[name]] in already:
                positions[name] += 1
            if positions[name] < len(group):
                frame_id = group[positions[name]]
                positions[name] += 1
                selected.append(frame_id)
                already.add(frame_id)
                progressed = True
            if len(selected) == size:
                break
        if not progressed:
            raise ValueError("Cannot fill training batch")
    if already & val_ids:
        raise AssertionError("Validation frame selected")

    model = json.loads(evidence.read_text(encoding="utf-8"))["model"]
    frames = []
    for frame_id in selected:
        sources = {part: f"{part}/{frame_id}.{suffix}" for part, suffix in (
            ("label_2", "txt"), ("calib", "txt"),
            ("image_2", "png"), ("velodyne", "bin"))}
        hashes = {part: file_sha256(dataset / relative)
                  for part, relative in sources.items()}
        before, after = frame_gt[frame_id]
        frames.append({
            "frame_id": frame_id,
            "car_gt_before_range_filter": before,
            "car_gt_after_range_filter": after,
            "stratum": car_group(after),
            "input_files": sources,
            "input_sha256": hashes,
        })
    return {
        "schema_version": "clocs_train_export_batch_v1",
        "status": "INPUTS_VERIFIED_EXPORT_PENDING",
        "split": "train",
        "sampling": "pilot: seeded SHA-256 ordering and round-robin GT-count strata",
        "sampling_seed": seed,
        "requested_size": size,
        "included_frames": anchors,
        "train_frames_scanned": len(train_ids),
        "train_pool_by_stratum": {name: len(pools[name]) for name in GROUPS},
        "split_manifest_sha256": file_sha256(manifest),
        "train_split_sha256": file_sha256(train),
        "validation_split_sha256": file_sha256(val),
        "pointpillars_checkpoint_sha256": model["checkpoint_sha256"],
        "pointpillars_config": model["config"],
        "mmdetection3d_commit": model["repository_commit"],
        "selected_ids_sha256": hashlib.sha256(
            ("\n".join(selected) + "\n").encode("ascii")).hexdigest(),
        "frames": frames,
        "interpretation": (
            "Train-only input/export pilot with deliberate GT-count balancing; "
            "not a representative corpus, a fusion result, or KITTI AP."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--manifest", default="configs/splits/kitti_second_split.json")
    parser.add_argument("--evidence", default="results/pointpillars_evidence_manifest.json")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--output", default="results/clocs_train_export_batch_manifest.json")
    args = parser.parse_args()
    result = build_export_batch(args.dataset, args.train, args.val, args.manifest,
                                args.evidence, args.size, args.seed, args.include)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "train_frames_scanned": result["train_frames_scanned"],
        "train_pool_by_stratum": result["train_pool_by_stratum"],
        "selected": [{"frame_id": frame["frame_id"],
                      "car_gt_after_range_filter": frame["car_gt_after_range_filter"]}
                     for frame in result["frames"]],
        "selected_ids_sha256": result["selected_ids_sha256"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
