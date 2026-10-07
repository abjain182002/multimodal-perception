from pathlib import Path
import csv

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")

CSV_PATH = (
    RESULTS_DIR
    / "kitti/camera/lidar_depth_comparison/yolov8_yolo26_lidar_comparison.csv"
)

FIGURE_PATH = (
    RESULTS_DIR
    / "kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png"
)


def load_results():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    numeric_fields = [
        "ground_truth_cars",
        "camera_boxes",
        "fused_detections",
        "matched_fused_cars",
        "mae_m",
        "median_ae_m",
        "rmse_m",
        "mean_signed_error_m",
    ]

    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])

    return rows


def make_figure(rows):
    labels = [
        f"{row['split']}\n{row['detector']}"
        for row in rows
    ]

    x = list(range(len(rows)))
    width = 0.22

    metrics = [
        ("mae_m", "MAE", -width),
        ("median_ae_m", "Median AE", 0.0),
        ("rmse_m", "RMSE", width),
    ]

    plt.figure(figsize=(10, 6))

    for field, label, offset in metrics:
        positions = [
            value + offset
            for value in x
        ]

        bars = plt.bar(
            positions,
            [row[field] for row in rows],
            width=width,
            label=label,
        )

        plt.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Depth Error (m)")

    plt.title(
        "KITTI Camera-LiDAR Depth Comparison: "
        "YOLOv8x vs YOLO26x"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = load_results()

    make_figure(rows)

    print("Comparison figure generated.")
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()

