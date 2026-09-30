"""Verify PV-RCNN evidence and render the frozen KITTI detector comparison."""
import argparse
import importlib.util
import subprocess
import sys
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pv_rcnn_matplotlib")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    for dependency in ("matplotlib", "numpy"):
        if importlib.util.find_spec(dependency) is None:
            raise RuntimeError(
                "Plotting dependency missing: " + dependency
                + ". Use an environment containing NumPy and Matplotlib before recording."
            )
    root = Path(__file__).resolve().parents[1]
    out = root / "results/pv_rcnn_kitti"
    expected = "709f818e3528acbb41e47cbb59fb0d541a188ac324d0269b7dd02ded017813a1"
    if digest(args.archive.read_bytes()) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(args.archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        manifest = json.loads(zipped.read("evidence_manifest.json"))
        if manifest["status"] != "PASS" or manifest["frames"] != 3769:
            raise ValueError("Invalid evidence manifest")
        if set(names) != set(manifest["members"]) | {"evidence_manifest.json"}:
            raise ValueError("Unexpected or missing archive members")
        for name, record in manifest["members"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe archive path")
            data = zipped.read(name)
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise ValueError("Archive member integrity mismatch: " + name)
        native = json.loads(zipped.read("results/native_ap40.json"))
        common = json.loads(zipped.read("results/common_ap40.json"))
        inference = json.loads(zipped.read("results/inference_audit.json"))
        if common["status"] != "PASS" or common["frames"] != 3769:
            raise ValueError("Common evaluator did not pass")
        if native["status"] != "NATIVE_EVALUATION_PASS":
            raise ValueError("Native evaluator did not pass")
        if inference["status"] != "INFERENCE_COMPLETE":
            raise ValueError("Incomplete inference")
        if common["maximum_absolute_ap_difference"] != 0:
            raise ValueError("Evaluator parity is not exact")
        if common["ap40_strict_iou_070"] != native["ap40_strict_iou_070"]:
            raise ValueError("Native/common AP mismatch")
        for key, member in (("predictions_sha256", "results/predictions.pkl"),
                            ("metadata_sha256", "metadata/kitti_infos_val.pkl")):
            if common[key] != native[key] or common[key] != manifest["members"][member]["sha256"]:
                raise ValueError("Prediction/metadata provenance mismatch")
        checkpoint = "08552932026855bcc10f66dbc45409f139d3208fd0095c2aab1f459f055fe463"
        if common["checkpoint_sha256"] != checkpoint:
            raise ValueError("Unexpected checkpoint")
        if inference["protocol"]["checkpoint_sha256"] != checkpoint:
            raise ValueError("Inference checkpoint differs")
        baseline_path = root / "results/clocs_full_val_reduced/ap40.json"
        baseline = json.loads(baseline_path.read_text())
        if baseline["status"] != "PASS" or baseline["frames"] != 3769:
            raise ValueError("CLOCs baseline-matched result unavailable")
        if baseline["baseline_max_absolute_ap40_error"] != 0:
            raise ValueError("PointPillars baseline parity did not pass")
        metadata_audit = json.loads(zipped.read("metadata/metadata_audit.json"))
        if metadata_audit["input_manifest_sha256"] != baseline["source_sha256"]["input_manifest"]:
            raise ValueError("PV-RCNN and PointPillars/CLOCs input manifests differ")
        if metadata_audit["metadata_sha256"] != common["metadata_sha256"]:
            raise ValueError("Metadata audit differs")
        ids = zipped.read("metadata/ImageSets/val.txt").decode().split()
        if len(ids) != len(set(ids)) or len(ids) != 3769:
            raise ValueError("Invalid validation IDs")
        # No pickle execution: metadata and predictions are verified as bytes.
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            destination = args.artifact_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = zipped.read(name)
            if destination.exists() and digest(destination.read_bytes()) != digest(data):
                raise ValueError("Existing artifact differs: " + str(destination))
            destination.write_bytes(data)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("common_ap40.json", "native_ap40.json", "inference_audit.json",
                     "protocol.json", "environment.txt"):
            (out / name).write_bytes(zipped.read("results/" + name))
        (out / "metadata_audit.json").write_bytes(zipped.read("metadata/metadata_audit.json"))
    receipt = {
        "status": "PASS", "frames": 3769, "archive_sha256": expected,
        "archive_bytes": args.archive.stat().st_size,
        "members_verified": len(manifest["members"]),
        "interpretation": "Official pretrained PV-RCNN validation; exact native/common evaluator parity."
    }
    (out / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "-m", "scripts.plot_pv_rcnn_results"],
        cwd=root, check=True,
    )
    print(json.dumps({"status": "PASS", "frames": 3769, "output": str(out),
                      "bulk_artifacts": str(args.artifact_dir)}, indent=2))


if __name__ == "__main__":
    main()
