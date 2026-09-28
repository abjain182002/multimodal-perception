#!/usr/bin/env python3
"""Run a resumable PointPillars dense export for any verified train manifest.

The manifest selects and hashes the source inputs.  Each candidate NPZ is
written atomically and independently verified, so a runtime disconnect can be
resumed without repeating completed frames or silently overwriting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


FIELDS = {
    "boxes_lidar": (107136, 7),
    "boxes_camera": (107136, 7),
    "scores_3d": (107136,),
    "direction_labels": (107136,),
    "projected_boxes": (107136, 4),
    "distance_82": (107136,),
    "candidate_index": (107136,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate(path: Path, frame_id: str, checkpoint_sha: str, config_sha: str, split_sha: str):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["frame_id"].item()) != frame_id:
            raise ValueError(f"frame ID mismatch in {path}")
        if str(archive["checkpoint_sha256"].item()) != checkpoint_sha:
            raise ValueError(f"checkpoint provenance mismatch in {path}")
        if str(archive["config_sha256"].item()) != config_sha:
            raise ValueError(f"config provenance mismatch in {path}")
        if str(archive["split_manifest_sha256"].item()) != split_sha:
            raise ValueError(f"split provenance mismatch in {path}")
        for field, shape in FIELDS.items():
            if archive[field].shape != shape:
                raise ValueError(f"{frame_id}: {field} shape {archive[field].shape} != {shape}")
        if not np.array_equal(archive["candidate_index"], np.arange(107136)):
            raise ValueError(f"candidate index coverage failed for {frame_id}")
        scores = archive["scores_3d"]
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError(f"invalid scores for {frame_id}")
        return {"num_candidates": int(scores.size), "score_max": float(scores.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "clocs_train_export_batch_v1":
        raise ValueError("unexpected export manifest schema")
    if manifest.get("status") != "INPUTS_VERIFIED_EXPORT_PENDING":
        raise ValueError("manifest is not export-pending")
    frames = manifest.get("frames", [])
    requested = int(manifest.get("requested_size", 0))
    if len(frames) != requested or requested <= 0:
        raise ValueError("manifest frame count is inconsistent")

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    expected_checkpoint = evidence["model"]["checkpoint_sha256"]
    checkpoint_sha = sha256(Path(args.checkpoint))
    if checkpoint_sha != expected_checkpoint:
        raise ValueError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    config_sha = sha256(Path(args.config))
    split_sha = sha256(Path(args.split_manifest))
    if split_sha != manifest["split_manifest_sha256"]:
        raise ValueError("split manifest SHA mismatch")
    if sha256(Path(args.train)) != manifest["train_split_sha256"]:
        raise ValueError("train split SHA mismatch")
    if sha256(Path(args.val)) != manifest["validation_split_sha256"]:
        raise ValueError("validation split SHA mismatch")

    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code = Path(args.code).resolve() if args.code else Path(__file__).resolve().parents[1]
    records = []
    for number, frame in enumerate(frames, start=1):
        frame_id = frame["frame_id"]
        output = output_dir / f"{frame_id}_dense_candidates.npz"
        if output.is_file():
            details = verify_candidate(output, frame_id, checkpoint_sha, config_sha, split_sha)
            action = "REUSED"
        else:
            partial = output_dir / f".{frame_id}_dense_candidates.partial.npz"
            partial.unlink(missing_ok=True)
            command = [
                args.python, "-m", "scripts.export_clocs_dense_train_frame",
                "--dataset", str(dataset), "--frame-id", frame_id,
                "--config", str(Path(args.config).resolve()),
                "--checkpoint", str(Path(args.checkpoint).resolve()),
                "--train", str(Path(args.train).resolve()),
                "--val", str(Path(args.val).resolve()),
                "--manifest", str(Path(args.split_manifest).resolve()),
                "--evidence", str(Path(args.evidence).resolve()),
                "--output", str(partial),
            ]
            subprocess.run(command, check=True, cwd=code,
                           env={**os.environ, "MPLBACKEND": "Agg"})
            details = verify_candidate(partial, frame_id, checkpoint_sha, config_sha, split_sha)
            partial.replace(output)
            action = "EXPORTED"
        digest = sha256(output)
        records.append({"frame_id": frame_id, "action": action,
                        "candidate_sha256": digest, **details})
        audit = {"schema_version": "clocs_dense_export_batch_audit_v1",
                 "status": "PASS" if len(records) == requested else "IN_PROGRESS",
                 "manifest_sha256": sha256(manifest_path),
                 "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
                 "split_manifest_sha256": split_sha, "requested_frames": requested,
                 "completed_frames": len(records), "frames": records}
        partial_audit = output_dir / "dense_export_audit.json.partial"
        partial_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        partial_audit.replace(output_dir / "dense_export_audit.json")
        print(f"{number}/{requested} {frame_id} {action} {digest}", flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
