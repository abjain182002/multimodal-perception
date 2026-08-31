from pathlib import Path
import csv

import numpy as np
from PIL import Image
from ultralytics import YOLO


NUM_FRAMES = 100
IOU_THRESHOLD = 0.30

KITTI_ROOT = Path.home() / "Major_Project/datasets/KITTI/training"
OUTPUT_DIR = Path.home() / "Major_Project/model_outputs/depth_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_calibration(path):
    calib = {}

    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.strip().split(":", 1)

            if value.strip():
                calib[key] = np.array(
                    [float(x) for x in value.split()],
                    dtype=np.float64,
                )

    return calib


def project_lidar(points, calib, width, height):
    P2 = calib["P2"].reshape(3, 4)
    R0 = calib["R0_rect"].reshape(3, 3)
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4)
    Tr_4[:3, :] = Tr

    xyz_hom = np.hstack(
        [points[:, :3], np.ones((len(points), 1))]
    )

    camera_points = (
        R0_4 @ Tr_4 @ xyz_hom.T
    ).T

    front = camera_points[:, 2] > 0
    camera_points = camera_points[front]

    projected = (
        P2 @ camera_points.T
    ).T

    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]
    depth = camera_points[:, 2]

    valid = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    return u[valid], v[valid], depth[valid]


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def v1_depth(depths):
    if len(depths) < 3:
        return None

    return float(np.median(depths))


def v2_depth(depths):
    if len(depths) < 3:
        return None

    sorted_depths = np.sort(depths)

    gaps = np.diff(sorted_depths)
    split_indices = np.where(gaps > 1.5)[0] + 1

    clusters = np.split(
        sorted_depths,
        split_indices,
    )

    min_cluster_size = max(
        3,
        int(0.05 * len(sorted_depths)),
    )

    valid_clusters = [
        cluster
        for cluster in clusters
        if len(cluster) >= min_cluster_size
    ]

    if not valid_clusters:
        return None

    foreground = min(
        valid_clusters,
        key=lambda c: np.median(c),
    )

    return float(np.median(foreground))


def load_ground_truth(path):
    gt = []

    with open(path) as f:
        for line in f:
            p = line.strip().split()

            if p[0] != "Car":
                continue

            gt.append({
                "bbox": np.array(
                    list(map(float, p[4:8]))
                ),
                "depth": float(p[13]),
                "truncation": float(p[1]),
                "occlusion": int(p[2]),
            })

    return gt


def process_frame(frame, model):
    image_path = KITTI_ROOT / "image_2" / f"{frame}.png"
    lidar_path = KITTI_ROOT / "velodyne" / f"{frame}.bin"
    calib_path = KITTI_ROOT / "calib" / f"{frame}.txt"
    label_path = KITTI_ROOT / "label_2" / f"{frame}.txt"

    image = np.array(Image.open(image_path))
    height, width = image.shape[:2]

    points = np.fromfile(
        lidar_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    calib = load_calibration(calib_path)

    u, v, depth = project_lidar(
        points,
        calib,
        width,
        height,
    )

    gt = load_ground_truth(label_path)

    result = model.predict(
        source=str(image_path),
        device="mps",
        conf=0.25,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        cls_id = int(box.cls.item())

        if model.names[cls_id] != "car":
            continue

        bbox = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = bbox

        inside = (
            (u >= x1)
            & (u <= x2)
            & (v >= y1)
            & (v <= y2)
        )

        depths = depth[inside]

        d1 = v1_depth(depths)
        d2 = v2_depth(depths)

        if d1 is None:
            continue

        detections.append({
            "bbox": bbox,
            "confidence": confidence,
            "v1": d1,
            "v2": d2,
            "points": len(depths),
        })

    candidates = []

    for det_idx, det in enumerate(detections):
        for gt_idx, truth in enumerate(gt):
            score = iou(
                det["bbox"],
                truth["bbox"],
            )

            candidates.append(
                (score, det_idx, gt_idx)
            )

    candidates.sort(reverse=True)

    used_dets = set()
    used_gt = set()
    rows = []

    for score, det_idx, gt_idx in candidates:
        if score < IOU_THRESHOLD:
            continue

        if det_idx in used_dets or gt_idx in used_gt:
            continue

        used_dets.add(det_idx)
        used_gt.add(gt_idx)

        det = detections[det_idx]
        truth = gt[gt_idx]

        gt_depth = truth["depth"]

        rows.append({
            "frame": frame,
            "iou": score,
            "confidence": det["confidence"],
            "gt_depth": gt_depth,
            "truncation": truth["truncation"],
            "occlusion": truth["occlusion"],
            "v1_depth": det["v1"],
            "v1_error": abs(det["v1"] - gt_depth),
            "v2_depth": det["v2"],
            "v2_error": (
                abs(det["v2"] - gt_depth)
                if det["v2"] is not None
                else None
            ),
            "lidar_points": det["points"],
        })

    return rows


def main():
    label_files = sorted(
        (KITTI_ROOT / "label_2").glob("*.txt")
    )

    all_frames = [
        path.stem
        for path in label_files
    ]

    # Spread the initial test across the KITTI training set.
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

    print("Loading YOLOv8x...")
    model = YOLO("yolov8x.pt")

    print()
    print(
        f"Evaluating {len(selected_frames)} KITTI frames"
    )
    print("=" * 70)

    all_rows = []

    for index, frame in enumerate(
        selected_frames,
        start=1,
    ):
        rows = process_frame(
            frame,
            model,
        )

        all_rows.extend(rows)

        print(
            f"[{index:02d}/{len(selected_frames):02d}] "
            f"frame={frame} "
            f"matches={len(rows)}"
        )

    print()
    print("=" * 70)

    if not all_rows:
        print("No matched objects found.")
        return

    v1_errors = np.array(
        [row["v1_error"] for row in all_rows]
    )

    v2_errors = np.array(
        [
            row["v2_error"]
            for row in all_rows
            if row["v2_error"] is not None
        ]
    )

    print(f"Frames evaluated: {len(selected_frames)}")
    print(f"Matched cars:     {len(all_rows)}")
    print()
    print(f"V1 MAE:           {np.mean(v1_errors):.3f} m")
    print(f"V1 Median AE:     {np.median(v1_errors):.3f} m")
    print(
        f"V1 RMSE:          "
        f"{np.sqrt(np.mean(v1_errors ** 2)):.3f} m"
    )

    if len(v2_errors):
        print()
        print(f"V2 MAE:           {np.mean(v2_errors):.3f} m")
        print(f"V2 Median AE:     {np.median(v2_errors):.3f} m")
        print(
            f"V2 RMSE:          "
            f"{np.sqrt(np.mean(v2_errors ** 2)):.3f} m"
        )

    csv_path = (
        OUTPUT_DIR
        / f"depth_validation_multiframe_visibility_{NUM_FRAMES}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
