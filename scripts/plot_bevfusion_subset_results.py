"""Reproduce figures from the transcribed paired subset summary (matplotlib)."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bevfusion_nuscenes_subset_20261002"


def main():
    data = json.loads((OUT / "evaluation_summary.json").read_text())
    assert data["samples"] == 1231 and data["scenes"] == 31
    assert sum(data["filtered_gt_counts"].values()) == 32276
    models = data["models"]
    colors = ["#245b86", "#d47725"]
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "bevfusion-subset-20261002"})
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label) in enumerate([("lidar", "LiDAR"), ("lidar_camera", "Camera + LiDAR")]):
        values = [models[key][m] * 100 for m in ["mAP", "NDS"]]
        bars = ax.bar([x + (i - 0.5) * 0.3 for x in [0, 1]], values,
                      width=0.3, label=label, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        for m in ["mAP", "NDS"]:
            rows.append([key, m, models[key][m], models[key][m] * 100])
    ax.set(xticks=[0, 1], xticklabels=["mAP", "NDS"], ylim=(0, 100),
           ylabel="Official metric × 100",
           title="BEVFusion pretrained pipelines: frozen nuScenes subset")
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.02, "1,231 samples · 31 scenes · availability-selected · not full validation",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "comparison")
    plt.close(fig)

    counts = data["filtered_gt_counts"]
    ordered = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh([k.replace("_", " ") for k in ordered],
                   [counts[k] for k in ordered], color=colors[0])
    ax.bar_label(bars, labels=[f"{counts[k]:,}" for k in ordered], padding=4)
    ax.set(xlim=(0, 13000), xlabel="Ground-truth boxes after official evaluation filters",
           title="Class support in the same 1,231-sample subset")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.02, "32,276 filtered boxes · all ten classes represented · counts are not accuracy",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "class_support")
    plt.close(fig)

    with (OUT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pipeline", "metric", "value_0_to_1", "value_times_100"])
        writer.writerows(rows)
        for metric in ["mAP", "NDS"]:
            delta = models["lidar_camera"][metric] - models["lidar"][metric]
            writer.writerow(["camera_lidar_minus_lidar", metric, delta, delta * 100])
    names = ["comparison.csv"] + [f"{n}.{ext}" for n in ["comparison", "class_support"]
                                   for ext in ["svg", "png", "pdf"]]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"script_sha256": digest(Path(__file__)),
                "input_sha256": digest(OUT / "evaluation_summary.json"),
                "artifacts": {name: digest(OUT / name) for name in names}}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("FIGURES AND COMPARISON PASS")


def save(fig, name):
    for ext in ["svg", "png", "pdf"]:
        kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(OUT / f"{name}.{ext}", **kwargs)


if __name__ == "__main__":
    main()
