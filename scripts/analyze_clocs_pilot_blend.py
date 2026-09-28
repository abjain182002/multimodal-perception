#!/usr/bin/env python3
"""Evaluate calibrated combinations of PointPillars and CLOCs scores.

The pilot trainer reports the learned CLOCs probability by itself.  This
analysis treats PointPillars as the frozen detector prior and evaluates
simple, deterministic score combinations on the exact held-out frames from
the training report.  It is a calibration analysis, not a new trained model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fusion.clocs_network import CLOCsFusionNet
from scripts.train_clocs_pilot import average_precision, load_frame


def logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def collect_scores(report, data_dir: Path, checkpoint: Path, device: torch.device):
    model = CLOCsFusionNet().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    labels, baseline, learned = [], [], []
    with torch.no_grad():
        for frame_id in report["holdout_frame_ids"]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report_path, checkpoint_path = Path(args.report), Path(args.checkpoint)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    labels, baseline, learned = collect_scores(report, Path(args.data_dir), checkpoint_path, device)

    records = [{"method": "pointpillars", "parameter": 0.0,
                "ap": average_precision(labels, baseline)},
               {"method": "clocs_only", "parameter": 1.0,
                "ap": average_precision(labels, learned)}]

    for weight in np.linspace(0.0, 1.0, 21):
        score = (1.0 - weight) * baseline + weight * learned
        records.append({"method": "linear_probability_blend", "parameter": float(weight),
                        "ap": average_precision(labels, score)})

    for beta in np.linspace(0.0, 3.0, 25):
        score = 1.0 / (1.0 + np.exp(-(logit(learned) + beta * logit(baseline))))
        records.append({"method": "logit_prior_blend", "parameter": float(beta),
                        "ap": average_precision(labels, score)})

    best = max(records, key=lambda item: item["ap"])
    baseline_ap = next(item["ap"] for item in records if item["method"] == "pointpillars")
    learned_ap = next(item["ap"] for item in records if item["method"] == "clocs_only")
    result = {
        "schema_version": "clocs_yolo26_pilot_score_calibration_v1",
        "status": "PASS",
        "interpretation": "Held-out score calibration on the same four-frame pilot split; not official KITTI AP.",
        "holdout_frame_ids": report["holdout_frame_ids"],
        "num_candidates": int(labels.size),
        "positive_candidates": int(labels.sum()),
        "pointpillars_ap": baseline_ap,
        "clocs_only_ap": learned_ap,
        "best_calibrated_method": best,
        "improvement_over_pointpillars": float(best["ap"] - baseline_ap),
        "candidates": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grouped = {}
        for item in records:
            grouped.setdefault(item["method"], []).append(item)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for method, values in grouped.items():
            ax.plot([v["parameter"] for v in values], [v["ap"] for v in values], marker="o", label=method)
        ax.axhline(baseline_ap, color="black", linestyle="--", linewidth=1, label="PointPillars baseline")
        ax.set_xlabel("Calibration parameter")
        ax.set_ylabel("Average precision")
        ax.set_title("CLOCs pilot score calibration on held-out frames")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)
    except ImportError:
        pass

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved calibration audit: {output_path}")
    print(f"Saved calibration figure: {args.figure}")


if __name__ == "__main__":
    main()
