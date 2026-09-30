"""Regenerate PV-RCNN comparison figures from committed verified audits."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pv_rcnn_matplotlib")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/pv_rcnn_kitti"
    baseline_path = root / "results/clocs_full_val_reduced/ap40.json"
    baseline = json.loads(baseline_path.read_text())
    common = json.loads((out / "common_ap40.json").read_text())
    native = json.loads((out / "native_ap40.json").read_text())
    receipt = json.loads((out / "archive_receipt.json").read_text())
    if not (baseline["status"] == common["status"] == receipt["status"] == "PASS"):
        raise ValueError("Verified results are required")
    if baseline["frames"] != 3769 or common["frames"] != 3769:
        raise ValueError("Unexpected frame count")
    if baseline["baseline_max_absolute_ap40_error"] != 0 or common["maximum_absolute_ap_difference"] != 0:
        raise ValueError("Baseline or evaluator parity failed")
    if common["ap40_strict_iou_070"] != native["ap40_strict_iou_070"]:
        raise ValueError("Native/common evaluator mismatch")
    expected = receipt["archive_sha256"]
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
            "plot_script": digest(Path(__file__).read_bytes()),
            "archive": expected,
        },
        "output_sha256": {p.name: digest(p.read_bytes()) for p in outputs},
    }
    (out / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2) + "\n")

    print("PV-RCNN comparison figures and source manifest: PASS")

if __name__ == "__main__":
    main()
