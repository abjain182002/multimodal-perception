"""Check preserved tables and build the supervisor follow-up's compact evidence.

No inference, physics simulation, raw-sensor access or official AP/NDS replay.
Run from any directory: python scripts/build_supervisor_followup.py
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/nuscenes/bevfusion/case_analysis/20261028"
ARM = ROOT / "experiments/robot_arm_pick_place/20261022"
OUT = ROOT / "results/supervisor_followup/20261028"


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def single(items, **where):
    matches = [r for r in items if all(r[k] == str(v) for k, v in where.items())]
    assert len(matches) == 1, where
    return matches[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame_path = CASE / "metadata_completion/results/ten_class_threshold_tradeoffs.csv"
    object_path = CASE / "results/object_evidence.csv"
    night_path = CASE / "night_cases/results/frame_threshold_diagnostics.csv"
    rain_path = CASE / "extension/results/real_rain_object_evidence.csv"
    visibility_path = CASE / "metadata_completion/results/focal_visibility.csv"
    sources = [frame_path, object_path, night_path, rain_path, visibility_path,
               CASE / "metadata_completion/results/completion_audit.json",
               ARM / "results/summary.csv", ARM / "results/trials.csv", ARM / "protocol.json"]
    frames, objects, night, rain, visibility = map(rows, sources[:5])
    assert (len(frames), len(objects), len(night), len(rain), len(visibility)) == (48, 16, 12, 4, 4)
    for r in frames + night:
        assert int(r["tp"]) + int(r["fn"]) == int(r["gt"])
        if r["precision"]:
            assert math.isclose(float(r["precision"]), int(r["tp"]) / (int(r["tp"]) + int(r["fp"])))
        if r["recall"]:
            assert math.isclose(float(r["recall"]), int(r["tp"]) / int(r["gt"]))
        if int(r["gt"]) == 0:
            assert not r["recall"], "Zero-GT recall must remain undefined"
    combined = []
    for case in "ABCD":
        for model, condition in [("lidar", "clean"), ("lidar_camera", "clean"),
                                 ("lidar_camera", "darkening_050"), ("lidar_camera", "darkening_025")]:
            f = single(frames, case=case, model=model, condition=condition, threshold="0.3")
            o = single(objects, case=case, model=model, condition=condition)
            v = single(visibility, case=case)
            combined.append({"case": case, "scene": o["scene"], "class": o["class_name"],
                             "model": model, "condition": condition, "threshold": 0.30,
                             "focal_score": o["nearest_score"], "focal_xy_error_m": o["nearest_xy_error_m"],
                             "focal_matched": o["matched_030"], "frame_gt": f["gt"],
                             "frame_tp": f["tp"], "frame_fp": f["fp"], "frame_fn": f["fn"],
                             "visibility_token": v["visibility_token"], "current_scan_lidar_points": v["annotation_lidar_points"],
                             "scope": "ten-class fixed-threshold diagnostics; not AP/NDS"})
    with (OUT / "case_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    b0 = single(objects, case="B", model="lidar_camera", condition="clean")
    b1 = single(objects, case="B", model="lidar_camera", condition="darkening_050")
    assert b0["matched_030"] == "False" and b1["matched_030"] == "True"
    assert float(b1["nearest_xy_error_m"]) > float(b0["nearest_xy_error_m"])
    completion = json.loads(sources[5].read_text())
    assert completion["B_half_vs_clean"]["gained_gt_indices"] == [26]
    assert completion["B_half_vs_clean"]["lost_gt_indices"] == []
    c0 = single(frames, case="C", model="lidar_camera", condition="clean", threshold="0.3")
    c1 = single(frames, case="C", model="lidar_camera", condition="darkening_025", threshold="0.3")
    assert (int(c0["tp"]), int(c1["tp"])) == (72, 56)
    for model in ("lidar", "lidar_camera"):
        n = single(night, scene="scene-1059", model=model, threshold="0.3")
        assert (int(n["tp"]), int(n["fp"]), int(n["fn"])) == (2, 0, 2)
        n = single(night, scene="scene-1060", model=model, threshold="0.3")
        assert (int(n["gt"]), int(n["fp"])) == (0, 1)

    protocol = json.loads((ARM / "protocol.json").read_text())
    trials, summary = rows(ARM / "results/trials.csv"), rows(ARM / "results/summary.csv")
    expected = {(p, q, str(s)) for p in protocol["profiles"] for q in protocol["policies"] for s in protocol["seeds"]}
    assert len(trials) == 45 and {(r["profile"], r["policy"], r["seed"]) for r in trials} == expected
    for profile in protocol["profiles"]:
        for seed in protocol["seeds"]:
            paired = [r for r in trials if r["profile"] == profile and r["seed"] == str(seed)]
            assert len({(r["offset_x_m"], r["offset_y_m"]) for r in paired}) == 1
    for r in summary:
        subset = [t for t in trials if t["profile"] == r["profile"] and t["policy"] == r["policy"]]
        assert len(subset) == int(r["trials"]) == 5
        assert sum(t["placed"] == "True" for t in subset) == int(r["placements"])
        assert math.isclose(mean(float(t["mean_normal_sum_N"]) for t in subset), float(r["mean_contact_load_N"]), abs_tol=1e-12)
        assert math.isclose(max(float(t["peak_normal_sum_N"]) for t in subset), float(r["max_contact_load_N"]), abs_tol=1e-12)
    high = single(summary, profile="heavy_component", policy="fixed_high")
    feedback = single(summary, profile="heavy_component", policy="slip_feedback")
    reduction = 100 * (1 - float(feedback["mean_contact_load_N"]) / float(high["mean_contact_load_N"]))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11.6, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 2.2])
    context = fig.add_subplot(grid[0, :]); context.axis("off")
    table = context.table(cellText=[["Light", "50 × 50 × 60 mm", "0.08 kg", "0.8"],
                                   ["Heavy", "50 × 50 × 60 mm", "0.50 kg", "0.4"],
                                   ["Narrow", "24 × 24 × 80 mm", "0.12 kg", "0.5"]],
                          colLabels=["Profile", "Object dimensions", "Mass", "Friction coefficient"],
                          loc="center", cellLoc="center", bbox=[0, 0.12, 1, 0.80])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d9de")
        if row == 0: cell.set_facecolor("#e6edf4"); cell.set_text_props(weight="bold")
    context.set_title("(a) Same light/heavy geometry; mass and friction both change", loc="left", pad=8)
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    colors = ["#737373", "#c87900", "#0072b2"]
    labels = ["Fixed low (3 N)", "Fixed high (20 N)", "Slip feedback (3–20 N)"]
    for i, (policy, label) in enumerate(zip(protocol["policies"], labels)):
        group = [single(summary, profile=p, policy=policy) for p in protocol["profiles"]]
        x = np.arange(3) + (i - 1) * 0.25
        bars = axes[0].bar(x, [int(r["placements"]) for r in group], width=0.23, color=colors[i], label=label)
        axes[0].bar_label(bars, labels=[f'{r["placements"]}/5' for r in group], padding=4, fontsize=9)
        bars = axes[1].bar(x, [float(r["mean_contact_load_N"]) for r in group], width=0.23, color=colors[i])
        axes[1].bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(3), ["Light", "Heavy", "Narrow"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.18)
    axes[0].set(title="(b) Placement outcome: show failed trials", ylabel="Completed placements out of five", ylim=(0, 6.1))
    axes[0].set_yticks(range(6))
    axes[1].set(title="(c) Equal success can hide different loading", ylabel="Mean summed normal contact force (N)", ylim=(0, 45))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False, title="Motor limit is per finger; contact force sums both fingers")
    fig.suptitle("Why similar movements are not equivalent handling outcomes\nSame commanded arm trajectory; recorded simulation results, no new trials", fontsize=14)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"arm_feedback_clarification.{suffix}", dpi=180,
                    metadata={"Creator": "Matplotlib"} if suffix != "png" else None)
    plt.close(fig)
    audit = {"status": "PRESERVED_TABLE_CHECKS_PASS", "source_snapshot": "31557146b6926aa1f6ee7089a32823bd9e3992b8",
             "case_threshold_rows_checked": len(frames), "night_threshold_rows_checked": len(night),
             "arm_trial_rows_reaggregated": len(trials), "arm_summary_groups_checked": len(summary),
             "heavy_feedback_mean_contact_reduction_percent": reduction,
             "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             "scope": "Preserved table consistency, pairing and arithmetic only. No inference, raw-prediction matching replay, official AP/NDS evaluation or physics rerun. Drive copy unverified."}
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
