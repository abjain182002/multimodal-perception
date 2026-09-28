#!/usr/bin/env python3
"""Select CLOCs score calibration on train frames and evaluate on holdout.

The exploratory calibration sweep selected its parameter on the holdout set.
This script removes that leakage: the grid parameter is selected using only
the optimization frames recorded by the pilot report, then evaluated once on
the fixed held-out frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fusion.clocs_network import CLOCsFusionNet
from scripts.train_clocs_pilot import average_precision, load_frame


def logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def collect_scores(report, frame_ids, data_dir, checkpoint, device):
    model = CLOCsFusionNet().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    labels, baseline, learned = [], [], []
    with torch.no_grad():
        for frame_id in frame_ids:
            frame = load_frame(data_dir, frame_id)
            example = frame["example"]
            features = torch.from_numpy(frame["features"]).to(device)
            indices = torch.from_numpy(frame["candidate_index"]).to(device)
            output = model(features, indices, int(example["num_candidates"]))
            labels_np = np.asarray(example["labels"], dtype=np.int8)
            valid = labels_np >= 0
            labels.append(labels_np[valid])
            baseline.append(np.asarray(example["candidate_scores_3d"], dtype=np.float64)[valid])
            learned.append(torch.sigmoid(output["candidate_logits"]).cpu().numpy()[valid])
    return np.concatenate(labels), np.concatenate(baseline), np.concatenate(learned)


def grid_records(labels, baseline, learned):
    records = []
    for weight in np.linspace(0.0, 1.0, 21):
        score = (1.0 - weight) * baseline + weight * learned
        records.append({"method": "linear_probability_blend", "parameter": float(weight),
                        "ap": average_precision(labels, score)})
    for beta in np.linspace(0.0, 3.0, 25):
        score = 1.0 / (1.0 + np.exp(-(logit(learned) + beta * logit(baseline))))
        records.append({"method": "logit_prior_blend", "parameter": float(beta),
                        "ap": average_precision(labels, score)})
    return records


def apply_choice(choice, baseline, learned):
    if choice["method"] == "linear_probability_blend":
        weight = choice["parameter"]
        return (1.0 - weight) * baseline + weight * learned
    beta = choice["parameter"]
    return 1.0 / (1.0 + np.exp(-(logit(learned) + beta * logit(baseline))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    train_labels, train_baseline, train_learned = collect_scores(
        report, report["train_frame_ids"], data_dir, checkpoint, device
    )
    holdout_labels, holdout_baseline, holdout_learned = collect_scores(
        report, report["holdout_frame_ids"], data_dir, checkpoint, device
    )

    train_grid = grid_records(train_labels, train_baseline, train_learned)
    selected = max(train_grid, key=lambda item: item["ap"])
    selected_holdout_score = apply_choice(selected, holdout_baseline, holdout_learned)
    train_baseline_ap = average_precision(train_labels, train_baseline)
    holdout_baseline_ap = average_precision(holdout_labels, holdout_baseline)
    train_selected_ap = selected["ap"]
    holdout_selected_ap = average_precision(holdout_labels, selected_holdout_score)

    result = {
        "schema_version": "clocs_yolo26_pilot_calibration_split_v1",
        "status": "PASS",
        "interpretation": "Calibration selected on 12 optimization frames and evaluated once on 4 fixed held-out frames; this remains a small pilot, not official KITTI AP.",
        "train_frame_ids": report["train_frame_ids"],
        "holdout_frame_ids": report["holdout_frame_ids"],
        "train_candidates": int(train_labels.size),
        "holdout_candidates": int(holdout_labels.size),
        "train_positive_candidates": int(train_labels.sum()),
        "holdout_positive_candidates": int(holdout_labels.sum()),
        "selected_on_train": selected,
        "train_baseline_pointpillars_ap": train_baseline_ap,
        "train_selected_calibrated_ap": train_selected_ap,
        "holdout_baseline_pointpillars_ap": holdout_baseline_ap,
        "holdout_selected_calibrated_ap": holdout_selected_ap,
        "holdout_improvement_over_pointpillars": holdout_selected_ap - holdout_baseline_ap,
        "train_grid": train_grid,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for method in ("linear_probability_blend", "logit_prior_blend"):
            values = [item for item in train_grid if item["method"] == method]
            axes[0].plot([item["parameter"] for item in values], [item["ap"] for item in values], marker="o", label=method)
        axes[0].axhline(train_baseline_ap, color="black", linestyle="--", label="PointPillars")
        axes[0].set_title("Train selection")
        axes[0].set_xlabel("Calibration parameter")
        axes[0].set_ylabel("AP")
        axes[0].grid(alpha=0.25)
        holdout_values = []
        for choice in train_grid:
            score = apply_choice(choice, holdout_baseline, holdout_learned)
            holdout_values.append({**choice, "ap": average_precision(holdout_labels, score)})
        for method in ("linear_probability_blend", "logit_prior_blend"):
            values = [item for item in holdout_values if item["method"] == method]
            axes[1].plot([item["parameter"] for item in values], [item["ap"] for item in values], marker="o", label=method)
        axes[1].axhline(holdout_baseline_ap, color="black", linestyle="--", label="PointPillars")
        axes[1].axvline(selected["parameter"], color="tab:red", linestyle=":", label="selected parameter")
        axes[1].set_title("Holdout evaluation")
        axes[1].set_xlabel("Calibration parameter")
        axes[1].set_ylabel("AP")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(args.figure, dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved split calibration audit: {output_path}")
    print(f"Saved split calibration figure: {args.figure}")


if __name__ == "__main__":
    main()
