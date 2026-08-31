from pathlib import Path
import json

import numpy as np


FRAME = "000149"
IOU_THRESHOLDS = [0.30, 0.50, 0.70]

FALCON_JSON = (
    Path.home()
    / "Major_Project/model_outputs/falcon_adapter"
    / f"{FRAME}_car.json"
)

KITTI_LABEL = (
    Path.home()
    / "Major_Project/datasets/KITTI/training/label_2"
    / f"{FRAME}.txt"
)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def load_falcon():
    with open(FALCON_JSON) as f:
        data = json.load(f)

    detections = []

    for detection in data["detections"]:
        box = detection["pixel_bbox"]

        detections.append(
            np.array([
                box["x1"],
                box["y1"],
                box["x2"],
                box["y2"],
            ])
        )

    return detections


def load_kitti():
    gt = []

    with open(KITTI_LABEL) as f:
        for line in f:
            parts = line.strip().split()

            if parts[0] != "Car":
                continue

            gt.append(
                np.array(
                    list(map(float, parts[4:8]))
                )
            )

    return gt


def main():
    detections = load_falcon()
    gt = load_kitti()

    candidates = []

    for det_idx, det_box in enumerate(detections):
        for gt_idx, gt_box in enumerate(gt):
            score = iou(det_box, gt_box)

            candidates.append(
                (score, det_idx, gt_idx)
            )

    candidates.sort(reverse=True)

    print()
    print("FALCON vs KITTI — FRAME", FRAME)
    print("=" * 78)
    print(f"Falcon detections: {len(detections)}")
    print(f"KITTI Car labels:  {len(gt)}")

    print()
    print("RESULTS BY IoU THRESHOLD")
    print("-" * 78)

    for threshold in IOU_THRESHOLDS:
        used_detections = set()
        used_gt = set()
        matches = []

        for score, det_idx, gt_idx in candidates:
            if score < threshold:
                continue

            if det_idx in used_detections:
                continue

            if gt_idx in used_gt:
                continue

            used_detections.add(det_idx)
            used_gt.add(gt_idx)

            matches.append(
                (score, det_idx, gt_idx)
            )

        tp = len(matches)
        fp = len(detections) - tp
        fn = len(gt) - tp

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        print(
            f"IoU >= {threshold:.2f} | "
            f"TP={tp:2d} | "
            f"FP={fp:2d} | "
            f"FN={fn:2d} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f}"
        )


if __name__ == "__main__":
    main()
