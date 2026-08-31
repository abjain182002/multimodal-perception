from pathlib import Path
import csv

import numpy as np


INPUT = (
    Path.home()
    / "Major_Project/model_outputs/depth_evaluation"
    / "depth_validation_multiframe_visibility_100.csv"
)


TRUNCATION_BINS = [
    (0.00, 0.15, "0.00-0.15"),
    (0.15, 0.30, ">0.15-0.30"),
    (0.30, 0.50, ">0.30-0.50"),
    (0.50, 1.01, ">0.50"),
]

OCCLUSION_NAMES = {
    0: "Fully visible",
    1: "Partly occluded",
    2: "Largely occluded",
    3: "Unknown",
}


def metrics(errors):
    errors = np.asarray(errors, dtype=float)

    return {
        "count": len(errors),
        "mae": np.mean(errors),
        "median": np.median(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
    }


def load_rows():
    rows = []

    with open(INPUT, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "frame": row["frame"],
                "gt_depth": float(row["gt_depth"]),
                "truncation": float(row["truncation"]),
                "occlusion": int(row["occlusion"]),
                "v1_error": float(row["v1_error"]),
                "v2_error": (
                    float(row["v2_error"])
                    if row["v2_error"].strip()
                    else None
                ),
            })

    return rows


def print_metrics(label, subset):
    if not subset:
        return

    v1 = metrics([
        row["v1_error"]
        for row in subset
    ])

    valid_v2 = [
        row["v2_error"]
        for row in subset
        if row["v2_error"] is not None
    ]

    print()
    print(label)
    print(f"  Cars:       {v1['count']}")
    print(f"  V1 MAE:     {v1['mae']:.3f} m")
    print(f"  V1 Median:  {v1['median']:.3f} m")
    print(f"  V1 RMSE:    {v1['rmse']:.3f} m")

    if valid_v2:
        v2 = metrics(valid_v2)

        print(f"  V2 MAE:     {v2['mae']:.3f} m")
        print(f"  V2 Median:  {v2['median']:.3f} m")
        print(f"  V2 RMSE:    {v2['rmse']:.3f} m")


def main():
    rows = load_rows()

    print()
    print("KITTI VISIBILITY ERROR ANALYSIS")
    print("=" * 72)
    print(f"Matched cars: {len(rows)}")

    print()
    print("ERROR BY TRUNCATION")
    print("-" * 72)

    for lower, upper, label in TRUNCATION_BINS:
        if lower == 0:
            subset = [
                row
                for row in rows
                if lower <= row["truncation"] <= upper
            ]
        else:
            subset = [
                row
                for row in rows
                if lower < row["truncation"] <= upper
            ]

        print_metrics(
            f"Truncation {label}",
            subset,
        )

    print()
    print("=" * 72)
    print("ERROR BY OCCLUSION")
    print("-" * 72)

    for code in sorted(OCCLUSION_NAMES):
        subset = [
            row
            for row in rows
            if row["occlusion"] == code
        ]

        print_metrics(
            f"{code} — {OCCLUSION_NAMES[code]}",
            subset,
        )


if __name__ == "__main__":
    main()
