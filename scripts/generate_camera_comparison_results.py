from pathlib import Path
import csv

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "kitti/camera/detector_comparison/camera_detector_comparison.csv"
FIGURE_PATH = RESULTS_DIR / "kitti/camera/detector_comparison/figures/camera_detector_comparison.png"


# Frozen results from the controlled KITTI pilot experiment.
#
# YOLOv8x:
#   confidence threshold = 0.25
#
# Falcon:
#   query = "car"
#   score-free IoU deduplication threshold = 0.50
#
# Both models:
#   same development frames
#   same independent holdout frames
#   same greedy one-to-one GT matching
#
# IMPORTANT:
# These are exploratory fixed-threshold metrics,
# NOT official KITTI AP/mAP results.

results = [
    # ---------------- DEVELOPMENT ----------------
    {
        "split": "Development",
        "model": "YOLOv8x",
        "iou_threshold": 0.30,
        "precision": 0.608,
        "recall": 0.938,
        "f1": 0.738,
        "mean_matched_iou": 0.710,
    },
    {
        "split": "Development",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.30,
        "precision": 0.526,
        "recall": 0.854,
        "f1": 0.651,
        "mean_matched_iou": 0.677,
    },
    {
        "split": "Development",
        "model": "YOLOv8x",
        "iou_threshold": 0.50,
        "precision": 0.541,
        "recall": 0.833,
        "f1": 0.656,
        "mean_matched_iou": 0.746,
    },
    {
        "split": "Development",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.50,
        "precision": 0.397,
        "recall": 0.646,
        "f1": 0.492,
        "mean_matched_iou": 0.749,
    },
    {
        "split": "Development",
        "model": "YOLOv8x",
        "iou_threshold": 0.70,
        "precision": 0.324,
        "recall": 0.500,
        "f1": 0.393,
        "mean_matched_iou": 0.838,
    },
    {
        "split": "Development",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.70,
        "precision": 0.269,
        "recall": 0.438,
        "f1": 0.333,
        "mean_matched_iou": 0.821,
    },

    # ---------------- HOLDOUT ----------------
    {
        "split": "Holdout",
        "model": "YOLOv8x",
        "iou_threshold": 0.30,
        "precision": 0.661,
        "recall": 0.911,
        "f1": 0.766,
        "mean_matched_iou": 0.767,
    },
    {
        "split": "Holdout",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.30,
        "precision": 0.583,
        "recall": 0.933,
        "f1": 0.718,
        "mean_matched_iou": 0.751,
    },
    {
        "split": "Holdout",
        "model": "YOLOv8x",
        "iou_threshold": 0.50,
        "precision": 0.581,
        "recall": 0.800,
        "f1": 0.673,
        "mean_matched_iou": 0.815,
    },
    {
        "split": "Holdout",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.50,
        "precision": 0.528,
        "recall": 0.844,
        "f1": 0.650,
        "mean_matched_iou": 0.787,
    },
    {
        "split": "Holdout",
        "model": "YOLOv8x",
        "iou_threshold": 0.70,
        "precision": 0.484,
        "recall": 0.667,
        "f1": 0.561,
        "mean_matched_iou": 0.856,
    },
    {
        "split": "Holdout",
        "model": "Falcon + Dedup",
        "iou_threshold": 0.70,
        "precision": 0.403,
        "recall": 0.644,
        "f1": 0.496,
        "mean_matched_iou": 0.840,
    },
]


def save_csv():
    fieldnames = [
        "split",
        "model",
        "iou_threshold",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def get_f1(split, model):
    rows = [
        row
        for row in results
        if row["split"] == split
        and row["model"] == model
    ]

    rows.sort(
        key=lambda row: row["iou_threshold"]
    )

    return [
        row["f1"]
        for row in rows
    ]


def make_figure():
    thresholds = [0.30, 0.50, 0.70]

    yolo_dev = get_f1(
        "Development",
        "YOLOv8x",
    )

    falcon_dev = get_f1(
        "Development",
        "Falcon + Dedup",
    )

    yolo_holdout = get_f1(
        "Holdout",
        "YOLOv8x",
    )

    falcon_holdout = get_f1(
        "Holdout",
        "Falcon + Dedup",
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        thresholds,
        yolo_dev,
        marker="o",
        label="YOLOv8x — Development",
    )

    plt.plot(
        thresholds,
        falcon_dev,
        marker="o",
        label="Falcon + Dedup — Development",
    )

    plt.plot(
        thresholds,
        yolo_holdout,
        marker="s",
        linestyle="--",
        label="YOLOv8x — Holdout",
    )

    plt.plot(
        thresholds,
        falcon_holdout,
        marker="s",
        linestyle="--",
        label="Falcon + Dedup — Holdout",
    )

    plt.xlabel("2D IoU Matching Threshold")
    plt.ylabel("F1 Score")

    plt.title(
        "KITTI Camera Detection Pilot: "
        "YOLOv8x vs Falcon Perception"
    )

    plt.xticks(thresholds)
    plt.ylim(0.0, 1.0)

    plt.grid(
        alpha=0.25
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

    save_csv()
    make_figure()

    print("Camera comparison results generated.")
    print()
    print("Saved CSV:")
    print(CSV_PATH)
    print()
    print("Saved figure:")
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()

