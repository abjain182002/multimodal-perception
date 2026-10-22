"""Validate paired records and regenerate a single report composite and table."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    rows = read("trials.csv")
    assert len(rows) == protocol["expected_trials"]
    keys = {(r["profile"], r["policy"], int(r["seed"])) for r in rows}
    expected = {(n, p, s) for n in protocol["profiles"] for p in protocol["policies"] for s in protocol["seeds"]}
    assert keys == expected
    for name in protocol["profiles"]:
        for seed in protocol["seeds"]:
            paired = [r for r in rows if r["profile"] == name and int(r["seed"]) == seed]
            assert len({(r["offset_x_m"], r["offset_y_m"]) for r in paired}) == 1
    summary = []
    for name in protocol["profiles"]:
        for policy in protocol["policies"]:
            subset = [r for r in rows if r["profile"] == name and r["policy"] == policy]
            for r in subset:
                trace = read(f'trace_{name}_{policy}_{r["seed"]}.csv')
                assert len(trace) == 300
                assert float(trace[-1]["time_s"]) == 15.0
                assert all(np.isfinite(float(t[k])) for t in trace for k in ["object_z_m", "ee_z_m", "normal_force_sum_N", "motor_limit_N"])
                assert all(3 <= float(t["motor_limit_N"]) <= 20 for t in trace)
                if policy != "slip_feedback":
                    assert {float(t["motor_limit_N"]) for t in trace} == {3.0 if policy == "fixed_low" else 20.0}
            successful = [float(r["xy_error_m"]) * 1000 for r in subset if r["placed"] == "True"]
            summary.append({"profile": name, "policy": policy, "trials": len(subset), "stable_lifts": sum(r["stable_lift"] == "True" for r in subset), "placements": sum(r["placed"] == "True" for r in subset), "mean_contact_load_N": float(np.mean([float(r["mean_normal_sum_N"]) for r in subset])), "max_contact_load_N": max(float(r["peak_normal_sum_N"]) for r in subset), "mean_final_motor_limit_N": float(np.mean([float(r["final_motor_limit_N"]) for r in subset])), "successful_mean_xy_error_mm": float(np.mean(successful)) if successful else None})
    with (RESULTS / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    tolerance_rows = []
    for name in protocol["profiles"]:
        for policy in protocol["policies"]:
            subset = [r for r in rows if r["profile"] == name and r["policy"] == policy]
            for tolerance_mm in (5, 10, 20, 30):
                tolerance_rows.append({"profile": name, "policy": policy, "xy_tolerance_mm": tolerance_mm, "placements": sum(r["placed"] == "True" and float(r["xy_error_m"]) < tolerance_mm / 1000 for r in subset), "trials": len(subset), "analysis": "post_hoc_tolerance_diagnostic"})
    with (RESULTS / "endpoint_tolerance.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=tolerance_rows[0]); w.writeheader(); w.writerows(tolerance_rows)
    table = ["| Component | Policy | Placements | Mean contact load (N) | Maximum contact load (N) |", "|---|---|---:|---:|---:|"]
    for r in summary:
        table.append(f'| {r["profile"]} | {r["policy"]} | {r["placements"]}/{r["trials"]} | {r["mean_contact_load_N"]:.2f} | {r["max_contact_load_N"]:.2f} |')
    (RESULTS / "summary.md").write_text("\n".join(table) + "\n\nContact load is the sum of simulated finger-object normal forces, averaged during gripping phases. Failed trials remain included; low load alone is not good performance. Five initial position perturbations per profile, not physical reliability estimates.\n")
    colors = ["#777777", "#e69f00", "#0072b2"]
    labels = ["Fixed low", "Fixed high", "Slip feedback"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(plt.imread(RESULTS / "heavy_component_slip_feedback_hold.png"))
    ax.axis("off")
    ax.set_title("(a) Panda arm: heavy component, seed 1", fontsize=11)
    names = protocol["profiles"]
    for col, (policy, label) in enumerate(zip(protocol["policies"], labels)):
        selected = [next(r for r in summary if r["profile"] == n and r["policy"] == policy) for n in names]
        x = np.arange(3) + (col - 1) * 0.24
        axes[0, 1].bar(x, [r["placements"] for r in selected], 0.22, color=colors[col], label=label)
        axes[1, 0].bar(x, [r["mean_contact_load_N"] for r in selected], 0.22, color=colors[col], label=label)
        trace = read(f"trace_heavy_component_{policy}_1.csv")
        axes[1, 1].plot([float(t["time_s"]) for t in trace], [float(t["object_z_m"]) * 100 for t in trace], color=colors[col], label=label)
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(3), ["Light", "Heavy", "Narrow"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 1].set(title="(b) Completed placements", ylabel="Successes out of 5")
    axes[0, 1].set_yticks(range(6))
    axes[0, 1].set_ylim(0, 6.2)
    axes[0, 1].legend(fontsize=8, loc="upper center", ncol=3)
    axes[1, 0].set(title="(c) Mean simulated contact loading", ylabel="Sum over both fingers (N)")
    axes[1, 1].set(title="(d) Heavy component: seed 1 trajectory", xlabel="Simulation time (s)", ylabel="Object centre height (cm)")
    axes[1, 1].axvspan(5, 10, color="#cccccc", alpha=0.15)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Factory pick-and-place pilot: ideal state feedback, rigid-body simulation", fontsize=13)
    fig.savefig(RESULTS / "arm_study_composite.png", dpi=160)
    fig.savefig(RESULTS / "arm_study_composite.pdf")
    plt.close(fig)
    audit = {"status": "PAIRED_SIMULATION_RECORDS_CHECKED", "trial_count": len(rows), "complete_protocol_keys": True, "paired_initial_offsets": True, "trace_duration_and_finiteness": True, "motor_limits_checked": True, "scope": "Record checks, not physical validation or verification of a perception pipeline."}
    (RESULTS / "record_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
