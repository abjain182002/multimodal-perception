from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


MODEL_OUTPUTS = (
    Path.home()
    / "Major_Project/model_outputs/depth_evaluation"
)

DEV_CSV = MODEL_OUTPUTS / "depth_validation_v3_100.csv"
HOLDOUT_CSV = MODEL_OUTPUTS / "depth_validation_v3_holdout_100.csv"

RESULTS_DIR = (
    Path.home()
    / "Major_Project/multimodal-perception/results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT = RESULTS_DIR / "kitti/depth/method_comparison/depth_method_comparison.csv"
FIGURE_OUTPUT = RESULTS_DIR / "kitti/depth/method_comparison/figures/depth_method_comparison.png"


def load_errors(path):
    data = {
        "V1": [],
        "V2": [],
        "V3": [],
    }

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data["V1"].append(float(row["v1_error"]))

            if row["v2_error"].strip():
                data["V2"].append(float(row["v2_error"]))

            data["V3"].append(float(row["v3_error"]))

    return data


def calculate_metrics(errors):
    values = np.asarray(errors, dtype=float)

    return {
        "count": len(values),
        "mae": float(np.mean(values)),
        "median_ae": float(np.median(values)),
        "rmse": float(
            np.sqrt(np.mean(values ** 2))
        ),
    }


def evaluate_split(name, path):
    errors = load_errors(path)

    rows = []

    for method in ["V1", "V2", "V3"]:
        metrics = calculate_metrics(errors[method])

        rows.append({
            "split": name,
            "method": method,
            "matched_objects": metrics["count"],
            "mae_m": metrics["mae"],
            "median_ae_m": metrics["median_ae"],
            "rmse_m": metrics["rmse"],
        })

    return rows


def save_table(rows):
    with open(TABLE_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "method",
                "matched_objects",
                "mae_m",
                "median_ae_m",
                "rmse_m",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows):
    methods = ["V1", "V2", "V3"]

    display_names = {
        "V1": "Median ROI",
        "V2": "Nearest Cluster",
        "V3": "Edge-Aware Hybrid",
    }

    dev = {
        row["method"]: row
        for row in rows
        if row["split"] == "Development"
    }

    holdout = {
        row["method"]: row
        for row in rows
        if row["split"] == "Holdout"
    }

    x = np.arange(len(methods))
    width = 0.36

    dev_mae = [
        dev[method]["mae_m"]
        for method in methods
    ]

    holdout_mae = [
        holdout[method]["mae_m"]
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_dev = ax.bar(
        x - width / 2,
        dev_mae,
        width,
        label="Development",
    )

    bars_holdout = ax.bar(
        x + width / 2,
        holdout_mae,
        width,
        label="Independent holdout",
    )

    ax.set_title(
        "KITTI Camera-LiDAR Depth Estimation: Development vs Holdout MAE"
    )
    ax.set_ylabel("Mean Absolute Error (m)")
    ax.set_xlabel("Depth Estimation Method")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_names[method] for method in methods]
    )
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in [bars_dev, bars_holdout]:
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(rows):
    print()
    print("DEPTH METHOD COMPARISON")
    print("=" * 76)

    for split in ["Development", "Holdout"]:
        print()
        print(split)
        print("-" * 76)

        for row in rows:
            if row["split"] != split:
                continue

            print(
                f"{row['method']} | "
                f"N={row['matched_objects']:3d} | "
                f"MAE={row['mae_m']:.3f} m | "
                f"Median AE={row['median_ae_m']:.3f} m | "
                f"RMSE={row['rmse_m']:.3f} m"
            )

    print()
    print("Final baseline decision:")
    print(
        "V1 retained because V3's development-set "
        "improvement did not generalize to holdout MAE/RMSE."
    )

    print()
    print("Saved:")
    print(TABLE_OUTPUT)
    print(FIGURE_OUTPUT)


def main():
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    dev_rows = evaluate_split(
        "Development",
        DEV_CSV,
    )

    holdout_rows = evaluate_split(
        "Holdout",
        HOLDOUT_CSV,
    )

    rows = dev_rows + holdout_rows

    save_table(rows)
    make_figure(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()

