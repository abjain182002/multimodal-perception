"""Compare original PointPillars KITTI predictions to the paired CLOCs run.

Only unpickle the trusted historical artifact whose SHA-256 is pinned below.
The pickle is a 3769-entry ordered list; sample_idx is the zero-based position,
not the six-digit KITTI frame ID. No AP is computed by this script.
"""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HISTORICAL_SHA256 = "2905f9c15fd04a6055dca99753b6fb6e6585492371608c5a625e3cc6ded764d3"
PAIRED_SHA256 = "1b034d06f49c343dfc4a21c80339481e37db455f237a19f3db3764d3a1306413"
FIELDS = ("name", "bbox", "dimensions", "location", "rotation_y", "score")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def compare_frame(original, current, index, fid):
    n_old = len(original["score"])
    n_new = len(current["score"])
    sample_ids = np.asarray(original["sample_idx"])
    if len(sample_ids) != n_old or np.any(sample_ids != index):
        raise ValueError(f"Historical sample_idx does not match position {index}: {fid}")
    if any(len(original[key]) != n_old or len(current[key]) != n_new
           for key in FIELDS):
        raise ValueError(f"Inconsistent annotation field lengths: {fid}")
    result = {"frame_id": fid, "index": index,
              "historical_count": n_old, "paired_count": n_new}
    if n_old != n_new:
        result["reason"] = "count"
        return result
    errors = {}
    for key in FIELDS:
        old = np.asarray(original[key])
        new = np.asarray(current[key])
        if key == "name":
            if not np.array_equal(old.astype(str), new.astype(str)):
                errors[key] = "different"
        elif old.shape != new.shape:
            errors[key] = "shape"
        elif old.size:
            delta = np.abs(old.astype(np.float64) - new.astype(np.float64))
            maximum = float(delta.max())
            if maximum > (1e-4 if key == "bbox" else 1e-5):
                errors[key] = maximum
    result["reason"] = "fields" if errors else "match"
    if errors:
        result["max_absolute_errors"] = errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--paired-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/clocs_full_val_paired/historical_prediction_comparison.json"))
    args = parser.parse_args()
    if sha256(args.historical) != HISTORICAL_SHA256:
        raise ValueError("Historical pickle SHA-256 mismatch; refusing to unpickle")
    if sha256(args.paired_archive) != PAIRED_SHA256:
        raise ValueError("Paired evidence ZIP SHA-256 mismatch")
    with args.historical.open("rb") as stream:
        historical = pickle.load(stream)
    if not isinstance(historical, list) or len(historical) != 3769:
        raise ValueError("Historical predictions must be an ordered 3769-entry list")
    with ZipFile(args.paired_archive) as archive:
        manifest = json.loads(archive.read("inputs/manifest.json"))
        ids = manifest["frame_ids"]
        if len(ids) != 3769 or len(set(ids)) != 3769:
            raise ValueError("Paired validation frame list changed")
        records = []
        for index, fid in enumerate(ids):
            frame = json.loads(archive.read(f"predictions/{fid}.json"))
            if frame["frame_id"] != fid:
                raise ValueError(f"Paired frame order mismatch at {index}")
            records.append(compare_frame(
                historical[index], frame["detections"]["pointpillars"], index, fid))
    reasons = Counter(record["reason"] for record in records)
    old_count = sum(record["historical_count"] for record in records)
    new_count = sum(record["paired_count"] for record in records)
    if old_count != 23844 or new_count != 24180:
        raise ValueError(f"Unexpected detection totals: {old_count}, {new_count}")
    audit = {
        "schema_version": "clocs_historical_prediction_comparison_v1",
        "status": "PASS",
        "interpretation": "Prediction artifact comparison only; does not establish evaluator or ground-truth parity.",
        "historical_sha256": HISTORICAL_SHA256,
        "paired_archive_sha256": PAIRED_SHA256,
        "frames": len(records),
        "detections": {"historical": old_count, "paired": new_count},
        "frame_comparison": dict(reasons),
        "changed_frames": [record for record in records if record["reason"] != "match"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "frames", "detections",
                                                 "frame_comparison")}, indent=2))
    print("First changed frames:", audit["changed_frames"][:8])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
