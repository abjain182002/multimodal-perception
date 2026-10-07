"""Render frozen baseline-matched CLOCs results; run from repository root."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/clocs_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/kitti/clocs/baseline_matched_validation"
    ap_path = out / "ap40.json"
    coverage_path = out / "coverage_diagnostic.json"
    ap = json.loads(ap_path.read_text())
    cov = json.loads(coverage_path.read_text())
    assert ap["status"] == cov["status"] == "PASS"
    assert ap["frames"] == cov["frames"] == 3769
    assert ap["baseline_max_absolute_ap40_error"] == 0
    metrics = ap["ap40_strict_iou_070"]
    difficulties = ("easy", "moderate", "hard")
    colors = ("#2864a0", "#df7926")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    x = np.arange(3)
    for col, metric in enumerate(("2D", "BEV", "3D")):
        baseline = np.array([metrics["pointpillars"][metric][d] for d in difficulties])
        fusion = np.array([metrics["fusion"][metric][d] for d in difficulties])
        ax = axes[0, col]
        for offset, values, color, label in ((-.19, baseline, colors[0], "PointPillars"),
                                            (.19, fusion, colors[1], "CLOCs + YOLO26x")):
            bars = ax.bar(x + offset, values, .38, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set(title=metric, ylim=(0, 105), xticks=x, xticklabels=["Easy", "Moderate", "Hard"])
        ax.set_ylabel("Car AP40 (%)")
        delta = fusion - baseline
        ax = axes[1, col]
        bars = ax.bar(x, delta, color=["#238552" if v >= 0 else "#b74747" for v in delta])
        ax.bar_label(bars, labels=[f"{v:+.3f}" for v in delta], padding=4, fontsize=9)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(ylim=(-3.1, 1.6), xticks=x, xticklabels=["Easy", "Moderate", "Hard"])
        ax.set_ylabel("Fusion minus baseline (AP points)")
        for d, b, f in zip(difficulties, baseline, fusion):
            rows.append([metric, d, float(b), float(f), float(f-b)])
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("KITTI validation: 3,769 frames, Car IoU 0.70\nHistorical baseline AP40 reproduced exactly")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"matched_ap40.{suffix}", dpi=200)
    plt.close(fig)
    with (out / "matched_ap40.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "difficulty", "pointpillars_ap40", "fusion_ap40", "difference_ap_points"])
        writer.writerows(rows)
    groups = cov["groups"]
    sets = [("<20 m", "20–40 m", "40–60 m", "60+ m"),
            ("occlusion_0", "occlusion_1", "occlusion_2", "occlusion_3")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    coverage_rows = []
    for ax, keys, title in zip(axes, sets, ("Camera depth", "KITTI occlusion label")):
        x = np.arange(len(keys))
        for offset, method, color in ((-.19, "pointpillars", colors[0]), (.19, "fusion", colors[1])):
            values = [100 * groups[k][method] / groups[k]["cars"] for k in keys]
            bars = ax.bar(x + offset, values, .38, color=color, label=method)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        labels = [f"{k.replace('occlusion_', '')}\nn={groups[k]['cars']:,}" for k in keys]
        ax.set(title=title, ylim=(0, 105), xticks=x, xticklabels=labels)
        ax.set_ylabel("Cars with best 3D IoU ≥ 0.70 (%)")
        for k in keys:
            g = groups[k]
            assert g["fusion"] - g["pointpillars"] == g["rescued"] - g["lost"]
            coverage_rows.append([k, g["cars"], g["pointpillars"], g["fusion"], g["rescued"], g["lost"],
                                  100*g["pointpillars"]/g["cars"], 100*g["fusion"]/g["cars"]])
    assert all(sum(groups[k][field] for k in keys) == groups["all"][field]
               for keys in sets for field in ("cars", "pointpillars", "fusion", "rescued", "lost"))
    axes[0].legend(fontsize=8)
    fig.suptitle("All labelled Cars: coverage diagnostic\nNot official difficulty matching, one-to-one recall, or AP40")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"matched_coverage.{suffix}", dpi=200)
    plt.close(fig)
    with (out / "matched_coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["group", "cars", "pointpillars_covered", "fusion_covered", "rescued", "lost",
                         "pointpillars_percent", "fusion_percent"])
        writer.writerows(coverage_rows)
    captions = """AP40 figure: Baseline-matched KITTI SECOND validation on 3,769 frames, Car strict IoU 0.70. Top: AP40; bottom: paired differences. The historical PointPillars baseline reproduces exactly across all nine metrics. One trained fusion checkpoint; no confidence intervals or significance test.

Coverage figure: All 14,385 labelled Cars, grouped by camera depth and occlusion label. Coverage uses each GT object's best retained-detection 3D IoU >= 0.70; it does not apply official difficulty/ignore matching or one-to-one assignment. Occlusion 0/1/2/3 denotes fully visible/partly occluded/largely occluded/unknown. These diagnostics do not establish the cause of Hard AP degradation.
"""
    (out / "figure_captions.md").write_text(captions)
    files = [ap_path, coverage_path, Path(__file__), out / "archive_receipt.json"]
    files += [out / name for name in ("matched_ap40.png", "matched_ap40.pdf", "matched_ap40.csv",
                                     "matched_coverage.png", "matched_coverage.pdf", "matched_coverage.csv",
                                     "figure_captions.md")]
    manifest = {"status": "PASS", "frames": 3769, "sha256": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Figures, CSVs, captions and source manifest: PASS")


if __name__ == "__main__":
    main()

