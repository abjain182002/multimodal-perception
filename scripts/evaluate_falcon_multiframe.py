from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.mlx.batch_inference import (
    BatchInferenceEngine,
    process_batch_and_generate,
)


NUM_FRAMES = 10
QUERY = "car"
IOU_THRESHOLDS = [0.30, 0.50, 0.70]

KITTI_ROOT = (
    Path.home()
    / "Major_Project/datasets/KITTI/training"
)

OUTPUT_DIR = (
    Path.home()
    / "Major_Project/model_outputs/falcon_multiframe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def pair_bbox_entries(raw):
    bboxes = []
    current = {}

    for entry in raw:
        if not isinstance(entry, dict):
            continue

        current.update(entry)

        if all(
            key in current
            for key in ("x", "y", "h", "w")
        ):
            bboxes.append(dict(current))
            current = {}

    return bboxes


def bbox_to_pixels(bbox, width, height):
    cx = float(bbox["x"])
    cy = float(bbox["y"])
    bw = float(bbox["w"])
    bh = float(bbox["h"])

    return np.array([
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ])


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
    path = (
        KITTI_ROOT
        / "label_2"
        / f"{frame}.txt"
    )

    boxes = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()

            if parts[0] != "Car":
                continue

            boxes.append(
                np.array(
                    list(map(float, parts[4:8]))
                )
            )

    return boxes


def match_boxes(detections, ground_truth, threshold):
    candidates = []

    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(ground_truth):
            score = iou(det, gt)

            candidates.append(
                (score, det_idx, gt_idx)
            )

    candidates.sort(reverse=True)

    used_dets = set()
    used_gt = set()
    matches = []

    for score, det_idx, gt_idx in candidates:
        if score < threshold:
            continue

        if det_idx in used_dets:
            continue

        if gt_idx in used_gt:
            continue

        used_dets.add(det_idx)
        used_gt.add(gt_idx)

        matches.append(score)

    tp = len(matches)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn, matches


def main():
    label_files = sorted(
        (KITTI_ROOT / "label_2").glob("*.txt")
    )

    all_frames = [
        path.stem
        for path in label_files
    ]

    indices = np.linspace(
        0,
        len(all_frames) - 1,
        NUM_FRAMES,
        dtype=int,
    )

    selected_frames = [
        all_frames[i]
        for i in indices
    ]

    print("Loading Falcon Perception MLX model...")

    load_start = time.perf_counter()

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )

    engine = BatchInferenceEngine(
        model,
        tokenizer,
    )

    model_load_time = (
        time.perf_counter() - load_start
    )

    print(
        f"Model loaded in {model_load_time:.2f}s"
    )

    prompt = build_prompt_for_task(
        QUERY,
        "detection",
    )

    totals = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": [],
        }
        for threshold in IOU_THRESHOLDS
    }

    frame_rows = []
    generation_times = []

    print()
    print(
        f"Evaluating {NUM_FRAMES} KITTI frames"
    )
    print("=" * 82)

    for index, frame in enumerate(
        selected_frames,
        start=1,
    ):
        image_path = (
            KITTI_ROOT
            / "image_2"
            / f"{frame}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        batch = process_batch_and_generate(
            tokenizer,
            [(image, prompt)],
            max_length=model_args.max_seq_len,
            min_dimension=256,
            max_dimension=1024,
        )

        start = time.perf_counter()

        _, aux_outputs = engine.generate(
            tokens=batch["tokens"],
            pos_t=batch["pos_t"],
            pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            max_new_tokens=200,
            temperature=0.0,
            task="detection",
        )

        generation_time = (
            time.perf_counter() - start
        )

        generation_times.append(
            generation_time
        )

        raw_boxes = pair_bbox_entries(
            aux_outputs[0].bboxes_raw
        )

        detections = [
            bbox_to_pixels(
                box,
                width,
                height,
            )
            for box in raw_boxes
        ]

        gt = load_ground_truth(frame)

        row = {
            "frame": frame,
            "detections": len(detections),
            "gt_cars": len(gt),
            "generation_seconds": generation_time,
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

            key = str(threshold).replace(".", "")

            row[f"tp_{key}"] = tp
            row[f"fp_{key}"] = fp
            row[f"fn_{key}"] = fn

        frame_rows.append(row)

        print(
            f"[{index:02d}/{NUM_FRAMES:02d}] "
            f"frame={frame} | "
            f"Falcon={len(detections):2d} | "
            f"GT={len(gt):2d} | "
            f"time={generation_time:.2f}s"
        )

    print()
    print("=" * 82)
    print("AGGREGATE RESULTS")
    print("-" * 82)

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
            np.mean(data["ious"])
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
            f"Mean matched IoU={mean_iou:.3f}"
        )

    mean_generation = np.mean(
        generation_times
    )

    print()
    print(
        f"Mean generation time/frame: "
        f"{mean_generation:.3f}s"
    )

    print(
        f"Approx generation FPS: "
        f"{1.0 / mean_generation:.3f}"
    )

    csv_path = (
        OUTPUT_DIR
        / "falcon_kitti_10_frames.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=frame_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(frame_rows)

    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
