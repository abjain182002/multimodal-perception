from pathlib import Path
import csv
import time

import numpy as np
from ultralytics import YOLO


NUM_FRAMES = 10
CONF_THRESHOLD = 0.25
IOU_THRESHOLDS = [0.30, 0.50, 0.70]

KITTI_ROOT = (
    Path.home()
    / "Major_Project/datasets/KITTI/training"
)

OUTPUT_DIR = (
    Path.home()
    / "Major_Project/model_outputs/yolov8x_camera"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def load_ground_truth(frame):
    label_path = (
        KITTI_ROOT
        / "label_2"
        / f"{frame}.txt"
    )

    boxes = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()

            if parts[0] != "Car":
                continue

            boxes.append(
                np.array(
                    list(map(float, parts[4:8])),
                    dtype=float,
                )
            )

    return boxes


def match_boxes(detections, ground_truth, threshold):
    candidates = []

    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(ground_truth):
            candidates.append(
                (
                    iou(det, gt),
                    det_idx,
                    gt_idx,
                )
            )

    candidates.sort(reverse=True)

    used_dets = set()
    used_gt = set()
    matched_ious = []

    for score, det_idx, gt_idx in candidates:
        if score < threshold:
            continue

        if det_idx in used_dets:
            continue

        if gt_idx in used_gt:
            continue

        used_dets.add(det_idx)
        used_gt.add(gt_idx)

        matched_ious.append(score)

    tp = len(matched_ious)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn, matched_ious


def make_totals():
    return {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": [],
        }
        for threshold in IOU_THRESHOLDS
    }


def evaluate_split(
    split_name,
    selected_frames,
    model,
):
    totals = make_totals()
    frame_rows = []
    prediction_times = []

    print()
    print(
        f"Evaluating YOLOv8x — {split_name}"
    )
    print("=" * 86)

    for index, frame in enumerate(
        selected_frames,
        start=1,
    ):
        image_path = (
            KITTI_ROOT
            / "image_2"
            / f"{frame}.png"
        )

        start = time.perf_counter()

        result = model.predict(
            source=str(image_path),
            device="mps",
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        prediction_time = (
            time.perf_counter() - start
        )

        prediction_times.append(
            prediction_time
        )

        detections = []

        for box in result.boxes:
            cls_id = int(
                box.cls.item()
            )

            if model.names[cls_id] != "car":
                continue

            detections.append(
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(float)
            )

        gt = load_ground_truth(frame)

        row = {
            "split": split_name,
            "frame": frame,
            "detections": len(detections),
            "gt_cars": len(gt),
            "prediction_seconds": prediction_time,
        }

        for threshold in IOU_THRESHOLDS:
            tp, fp, fn, matched_ious = match_boxes(
                detections,
                gt,
                threshold,
            )

            totals[threshold]["tp"] += tp
            totals[threshold]["fp"] += fp
            totals[threshold]["fn"] += fn
            totals[threshold]["ious"].extend(
                matched_ious
            )

            key = str(
                threshold
            ).replace(".", "")

            row[f"tp_{key}"] = tp
            row[f"fp_{key}"] = fp
            row[f"fn_{key}"] = fn

        frame_rows.append(row)

        print(
            f"[{index:02d}/{len(selected_frames):02d}] "
            f"frame={frame} | "
            f"YOLO={len(detections):2d} | "
            f"GT={len(gt):2d} | "
            f"time={prediction_time:.3f}s"
        )

    print()
    print("-" * 86)

    summary = []

    for threshold in IOU_THRESHOLDS:
        data = totals[threshold]

        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        mean_iou = (
            float(np.mean(data["ious"]))
            if data["ious"]
            else 0.0
        )

        print(
            f"IoU >= {threshold:.2f} | "
            f"TP={tp:3d} | "
            f"FP={fp:3d} | "
            f"FN={fn:3d} | "
            f"P={precision:.3f} | "
            f"R={recall:.3f} | "
            f"F1={f1:.3f} | "
            f"Mean IoU={mean_iou:.3f}"
        )

        summary.append({
            "split": split_name,
            "iou_threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_iou,
        })

    mean_time = float(
        np.mean(prediction_times)
    )

    print()
    print(
        f"Mean prediction time/frame: "
        f"{mean_time:.3f}s"
    )

    print(
        f"Approx prediction FPS: "
        f"{1.0 / mean_time:.3f}"
    )

    return (
        frame_rows,
        summary,
        mean_time,
    )


def main():
    label_files = sorted(
        (KITTI_ROOT / "label_2").glob("*.txt")
    )

    all_frames = [
        path.stem
        for path in label_files
    ]

    # --------------------------------------------------
    # Same development frames used for Falcon
    # --------------------------------------------------
    development_indices = np.linspace(
        0,
        len(all_frames) - 1,
        NUM_FRAMES,
        dtype=int,
    )

    development_frames = [
        all_frames[i]
        for i in development_indices
    ]

    # --------------------------------------------------
    # Same deterministic independent holdout used for Falcon
    # --------------------------------------------------
    development_index_set = set(
        development_indices.tolist()
    )

    available_indices = np.array([
        i
        for i in range(len(all_frames))
        if i not in development_index_set
    ])

    rng = np.random.default_rng(42)

    holdout_indices = np.sort(
        rng.choice(
            available_indices,
            size=NUM_FRAMES,
            replace=False,
        )
    )

    holdout_frames = [
        all_frames[i]
        for i in holdout_indices
    ]

    print("Loading YOLOv8x...")

    model = YOLO(
        "yolov8x.pt"
    )

    dev_rows, dev_summary, dev_time = evaluate_split(
        "Development",
        development_frames,
        model,
    )

    holdout_rows, holdout_summary, holdout_time = evaluate_split(
        "Holdout",
        holdout_frames,
        model,
    )

    all_rows = (
        dev_rows
        + holdout_rows
    )

    all_summary = (
        dev_summary
        + holdout_summary
    )

    frame_csv = (
        OUTPUT_DIR
        / "yolov8x_kitti_frames.csv"
    )

    with open(
        frame_csv,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = (
        OUTPUT_DIR
        / "yolov8x_kitti_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_summary[0].keys(),
        )

        writer.writeheader()
        writer.writerows(all_summary)

    print()
    print("=" * 86)
    print("YOLOv8x evaluation complete")
    print(
        f"Development mean time: "
        f"{dev_time:.3f}s"
    )
    print(
        f"Holdout mean time:     "
        f"{holdout_time:.3f}s"
    )

    print()
    print("Saved:")
    print(frame_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()
