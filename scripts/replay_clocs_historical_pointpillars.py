"""Replay the verified historical PointPillars predictions against paired-run GT.

This evaluates existing predictions only; no model inference or fusion training.
Run in the pinned MMDetection3D environment with the NVIDIA Numba binding.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from scripts.evaluate_clocs_full_validation import BASELINE, array_annotation, evaluate
from scripts.run_clocs_full_validation import EXPECTED, sha256, verify_manifest

HISTORICAL_SHA256 = "2905f9c15fd04a6055dca99753b6fb6e6585492371608c5a625e3cc6ded764d3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("NUMBA_CUDA_USE_NVIDIA_BINDING") != "1" or os.environ.get("NUMBA_ENABLE_CUDASIM"):
        raise RuntimeError("Use the NVIDIA Numba binding without CUDA simulator")
    from importlib.metadata import version
    if version("cuda-python") != "12.2.1":
        raise RuntimeError("Expected cuda-python 12.2.1")
    if sha256(args.historical) != HISTORICAL_SHA256:
        raise ValueError("Historical pickle SHA-256 differs; refusing to unpickle")
    ids = verify_manifest(json.loads(args.manifest.read_text()))
    provenance = dict(input_manifest_sha256=sha256(args.manifest), **EXPECTED)
    with args.historical.open("rb") as stream:
        historical = pickle.load(stream)
    if not isinstance(historical, list) or len(historical) != len(ids):
        raise ValueError("Historical predictions must match 3769 ordered validation entries")

    ground_truth = []
    old_detections = []
    current_detections = []
    ordered = hashlib.sha256()
    for index, fid in enumerate(ids):
        raw = (args.predictions / f"{fid}.json").read_bytes()
        ordered.update(hashlib.sha256(raw).digest())
        frame = json.loads(raw)
        if frame.get("frame_id") != fid or frame.get("provenance") != provenance:
            raise ValueError(f"Paired prediction identity/provenance differs: {fid}")
        old = historical[index]
        if set(old) < {"name", "bbox", "dimensions", "location", "rotation_y", "score", "sample_idx"}:
            raise ValueError(f"Historical annotation missing fields: {index}")
        indices = np.asarray(old["sample_idx"])
        if len(indices) != len(old["score"]) or np.any(indices != index):
            raise ValueError(f"Historical sample index mismatch: {index} ({fid})")
        ground_truth.append(array_annotation(frame["ground_truth"], detection=False))
        old_detections.append(array_annotation(old, detection=True))
        current_detections.append(array_annotation(frame["detections"]["pointpillars"], detection=True))
    if (sum(len(x["score"]) for x in old_detections) != 23844 or
            sum(len(x["score"]) for x in current_detections) != 24180):
        raise ValueError("Expected historical and paired detection totals")

    print("INPUTS PASS: 3769 ordered frames; 23844 historical detections; 24180 paired detections",
          flush=True)
    historical_ap = evaluate(ground_truth, old_detections)
    differences = {
        metric: {
            difficulty: historical_ap[metric][difficulty] - reference
            for difficulty, reference in zip(("easy", "moderate", "hard"), values)
        } for metric, values in BASELINE.items()
    }
    largest = max(abs(value) for row in differences.values() for value in row.values())
    result = {
        "schema_version": "clocs_historical_pointpillars_replay_v1",
        "status": "HISTORICAL_AP_REPRODUCED" if largest <= .05 else "HISTORICAL_AP_MISMATCH",
        "interpretation": (
            "Historical saved PointPillars detections evaluated against ground truth "
            "from the paired run with pinned MMDetection3D. This isolates prediction "
            "versus ground-truth/evaluator differences; no fusion AP is recalculated."
        ),
        "frames": len(ids),
        "historical_prediction_sha256": HISTORICAL_SHA256,
        "paired_input_manifest_sha256": sha256(args.manifest),
        "paired_ordered_prediction_digests": ordered.hexdigest(),
        "historical_detections": 23844,
        "paired_detections": 24180,
        "historical_replay_ap40_strict_iou_070": historical_ap,
        "historical_reference_ap40_strict_iou_070": {
            metric: dict(zip(("easy", "moderate", "hard"), values))
            for metric, values in BASELINE.items()
        },
        "replay_minus_reference_ap_points": differences,
        "max_absolute_ap40_error": largest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
