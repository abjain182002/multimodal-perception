"""Verify PV-RCNN evidence and render the frozen KITTI detector comparison."""
import argparse
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
    methods = {
        "PointPillars": baseline["ap40_strict_iou_070"]["pointpillars"],
        "CLOCs + YOLO26x": baseline["ap40_strict_iou_070"]["fusion"],
        "PV-RCNN": common["ap40_strict_iou_070"],
    }
    rows = []
    differences = []
    for metric in ("2D", "BEV", "3D"):
        for difficulty in ("easy", "moderate", "hard"):
            b = methods["PointPillars"][metric][difficulty]
            for method, values in methods.items():
                value = values[metric][difficulty]
                rows.append([method, metric, difficulty, value, value-b])
            differences.append([metric, difficulty, methods["PV-RCNN"][metric][difficulty]-b])
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["method", "metric", "difficulty", "ap40", "minus_pointpillars_ap_points"])
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), constrained_layout=True)
    x = np.arange(3)
    for ax, metric in zip(axes, ("2D", "BEV", "3D")):
        for index, (method, values) in enumerate(methods.items()):
            bars = ax.bar(x + (index-1)*.26,
                          [values[metric][d] for d in ("easy", "moderate", "hard")],
                          .26, label=method, color=("#2864a0", "#df7926", "#398552")[index])
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
        ax.set(title=metric, ylim=(0, 105), xticks=x,
               xticklabels=["Easy", "Moderate", "Hard"], ylabel="Car AP40 (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("KITTI validation: 3,769 frames, Car IoU 0.70\nCommon evaluator; detector architectures and training differ")
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("comparison." + suffix), dpi=200)
    plt.close(fig)
    findings = """# PV-RCNN full KITTI validation — 30 September 2026

The official pretrained OpenPCDet PV-RCNN checkpoint scored 92.1047 / 84.3604 / 82.4830 Car 3D AP40 (Easy / Moderate / Hard) on the frozen 3,769-frame SECOND validation split at strict IoU 0.70. Native OpenPCDet and the pinned MMDetection3D evaluator agreed exactly across all nine 2D/BEV/3D metrics.

Relative to the verified PointPillars baseline, PV-RCNN gained 1.7598 / 5.2306 / 6.3265 3D AP points. This establishes a stronger LiDAR-only baseline in this experiment. It does not establish an advantage from fusion, equal training budgets, statistical significance or universal model superiority. Both detectors are pretrained; the CLOCs fusion head used 96 optimization and 32 development frames. Checkpoint-specific upstream training provenance should accompany the comparison.

The model is PV-RCNN, not PV-RCNN++ or Azitt. No PV-RCNN camera fusion, retraining, standardized runtime benchmark or cross-dataset evaluation has yet been measured. Inference progress timing is not a controlled runtime benchmark.

The implementation reused OpenPCDet commit 233f849829b6ac19afb8af8837a0246890908755 and its official KITTI model-zoo checkpoint. All 367/367 checkpoint parameters loaded. Camera-FOV filtering, voxelization and KITTI prediction conversion used upstream code. The local wrapper provides headless/resumable inference, integrity checks and cross-framework evaluator verification.

Setup used Python 3.10, Torch 2.1.2+cu121, NumPy 1.26.4 and spconv-cu120 2.3.6 on a Tesla T4. The installed wheel omitted Python helper folders; missing files were restored from the same pinned checkout. Kornia 0.6.7 supplies the API required by the Argoverse helper imported by OpenPCDet. See environment.txt for resolved packages.

Figure caption: Car strict IoU 0.70 AP40 on the same 3,769 KITTI validation frames with a common evaluator. PointPillars and CLOCs use the verified reduced-FOV run; PV-RCNN uses official OpenPCDet preprocessing. Architecture, training recipes, point-cloud ranges and postprocessing differ. These are complete-pipeline comparisons, not an isolated architecture ablation.

Sources: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755), [PV-RCNN paper, Shi et al., CVPR 2020](https://arxiv.org/abs/1912.13192). Reused code and pretrained weights must be attributed; these measurements are our reproduction, not copied model-zoo scores.
"""
    (out / "FINDINGS.md").write_text(findings)
    outputs = sorted(p for p in out.iterdir() if p.is_file() and p.name != "figure_manifest.json")
    figure_manifest = {
        "status": "PASS",
        "source_sha256": {
            "baseline_ap40": digest(baseline_path.read_bytes()),
            "recorder_script": digest(Path(__file__).read_bytes()),
            "archive": expected,
        },
        "output_sha256": {p.name: digest(p.read_bytes()) for p in outputs},
    }
    (out / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "frames": 3769, "output": str(out),
                      "pv_rcnn_3d": methods["PV-RCNN"]["3D"],
                      "bulk_artifacts": str(args.artifact_dir)}, indent=2))


if __name__ == "__main__":
    main()
