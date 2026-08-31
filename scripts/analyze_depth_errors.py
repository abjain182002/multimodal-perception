from pathlib import Path
import csv

import numpy as np


INPUT = (
    Path.home()
    / "Major_Project/model_outputs/depth_evaluation"
    / "depth_validation_multiframe_100.csv"
)

OUTPUT_DIR = (
    Path.home()
    / "Major_Project/model_outputs/depth_evaluation"
)

DISTANCE_BINS = [
    (0, 20, "0-20 m"),
    (20, 40, "20-40 m"),
    (40, 60, "40-60 m"),
    (60, float("inf"), "60+ m"),
]


def load_rows():
    rows = []

    with open(INPUT, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "frame": row["frame"],
                "iou": float(row["iou"]),
                "confidence": float(row["confidence"]),
                "gt_depth": float(row["gt_depth"]),
                "v1_depth": float(row["v1_depth"]),
                "v1_error": float(row["v1_error"]),
                "v2_depth": (
                    float(row["v2_depth"])
                    if row["v2_depth"].strip()
                    else None
                ),
                "v2_error": (
                    float(row["v2_error"])
                    if row["v2_error"].strip()
                    else None
                ),
                "lidar_points": int(row["lidar_points"]),
            })

    return rows


def metrics(errors):
    errors = np.array(errors, dtype=float)

    return {
        "count": len(errors),
        "mae": np.mean(errors),
        "median_ae": np.median(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
    }


def main():
    rows = load_rows()

    print()
    print("KITTI DEPTH ERROR ANALYSIS")
    print("=" * 78)
    print(f"Matched cars: {len(rows)}")

    # --------------------------------------------------
    # Overall signed bias
    # estimated - ground truth
    # --------------------------------------------------
    v1_signed = np.array([
        row["v1_depth"] - row["gt_depth"]
        for row in rows
    ])

    v2_signed = np.array([
        row["v2_depth"] - row["gt_depth"]
        for row in rows
        if row["v2_depth"] is not None
    ])

    print()
    print("SIGNED DEPTH BIAS")
    print("-" * 78)

    print(
        f"V1 mean signed error: {np.mean(v1_signed):.3f} m"
    )

    print(
        f"V2 mean signed error: {np.mean(v2_signed):.3f} m"
    )

    print()
    print(
        "Negative value = estimated depth is closer "
        "than KITTI object-center depth."
    )

    # --------------------------------------------------
    # Distance-bin analysis
    # --------------------------------------------------
    print()
    print("ERROR BY GROUND-TRUTH DISTANCE")
    print("-" * 78)

    summary_rows = []

    for lower, upper, label in DISTANCE_BINS:
        subset = [
            row
            for row in rows
            if lower <= row["gt_depth"] < upper
        ]

        if not subset:
            continue

        v1 = metrics([
            row["v1_error"]
            for row in subset
        ])

        valid_v2_errors = [
            row["v2_error"]
            for row in subset
            if row["v2_error"] is not None
        ]

        v2 = (
            metrics(valid_v2_errors)
            if valid_v2_errors
            else {
                "count": 0,
                "mae": float("nan"),
                "median_ae": float("nan"),
                "rmse": float("nan"),
            }
        )

        print()
        print(f"{label}")
        print(f"  Cars:       {len(subset)}")
        print(f"  V1 MAE:     {v1['mae']:.3f} m")
        print(f"  V1 Median:  {v1['median_ae']:.3f} m")
        print(f"  V1 RMSE:    {v1['rmse']:.3f} m")
        print(f"  V2 MAE:     {v2['mae']:.3f} m")
        print(f"  V2 Median:  {v2['median_ae']:.3f} m")
        print(f"  V2 RMSE:    {v2['rmse']:.3f} m")

        summary_rows.append({
            "distance_bin": label,
            "cars": len(subset),
            "v1_mae": v1["mae"],
            "v1_median_ae": v1["median_ae"],
            "v1_rmse": v1["rmse"],
            "v2_mae": v2["mae"],
            "v2_median_ae": v2["median_ae"],
            "v2_rmse": v2["rmse"],
        })

    summary_path = (
        OUTPUT_DIR
        / "depth_summary_by_distance_100.csv"
    )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------
    # Worst V1 outliers
    # --------------------------------------------------
    outliers = sorted(
        rows,
        key=lambda row: row["v1_error"],
        reverse=True,
    )[:10]

    print()
    print("=" * 78)
    print("TOP 10 V1 OUTLIERS")
    print("-" * 78)

    for i, row in enumerate(outliers, start=1):
        print(
            f"{i:02d} | "
            f"frame={row['frame']} | "
            f"GT={row['gt_depth']:.2f} m | "
            f"V1={row['v1_depth']:.2f} m | "
            f"error={row['v1_error']:.2f} m | "
            f"IoU={row['iou']:.3f} | "
            f"points={row['lidar_points']}"
        )

    outlier_path = (
        OUTPUT_DIR
        / "depth_outliers_100.csv"
    )

    with open(outlier_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=outliers[0].keys(),
        )
        writer.writeheader()
        writer.writerows(outliers)

    print()
    print("Saved:")
    print(summary_path)
    print(outlier_path)


if __name__ == "__main__":
    main()
