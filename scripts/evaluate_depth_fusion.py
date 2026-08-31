from pathlib import Path
import csv

import numpy as np
from PIL import Image
from ultralytics import YOLO


FRAME = "000149"

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

    xyz = points[:, :3]

    xyz_hom = np.hstack(
        [xyz, np.ones((len(xyz), 1))]
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

            bbox = np.array(
                list(map(float, p[4:8]))
            )

            z = float(p[13])

            gt.append({
                "bbox": bbox,
                "depth": z,
            })

    return gt


def main():
    image_path = KITTI_ROOT / "image_2" / f"{FRAME}.png"
    lidar_path = KITTI_ROOT / "velodyne" / f"{FRAME}.bin"
    calib_path = KITTI_ROOT / "calib" / f"{FRAME}.txt"
    label_path = KITTI_ROOT / "label_2" / f"{FRAME}.txt"

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

    model = YOLO("yolov8x.pt")

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

    # Greedy one-to-one matching by highest IoU.
    candidates = []

    for det_idx, det in enumerate(detections):
        for gt_idx, truth in enumerate(gt):
            score = iou(det["bbox"], truth["bbox"])

            candidates.append(
                (score, det_idx, gt_idx)
            )

    candidates.sort(reverse=True)

    used_dets = set()
    used_gt = set()
    matches = []

    for score, det_idx, gt_idx in candidates:
        if score < 0.3:
            continue

        if det_idx in used_dets or gt_idx in used_gt:
            continue

        used_dets.add(det_idx)
        used_gt.add(gt_idx)

        matches.append(
            (score, detections[det_idx], gt[gt_idx])
        )

    rows = []

    print()
    print("YOLOv8x + LiDAR depth validation")
    print("=" * 100)

    for i, (score, det, truth) in enumerate(matches, 1):
        gt_depth = truth["depth"]

        v1_error = abs(det["v1"] - gt_depth)

        v2_error = (
            abs(det["v2"] - gt_depth)
            if det["v2"] is not None
            else None
        )

        row = {
            "match": i,
            "iou": score,
            "confidence": det["confidence"],
            "gt_depth": gt_depth,
            "v1_depth": det["v1"],
            "v1_error": v1_error,
            "v2_depth": det["v2"],
            "v2_error": v2_error,
            "lidar_points": det["points"],
        }

        rows.append(row)

        print(
            f"{i:02d} | "
            f"IoU={score:.3f} | "
            f"conf={det['confidence']:.3f} | "
            f"GT={gt_depth:6.2f} m | "
            f"V1={det['v1']:6.2f} "
            f"(err {v1_error:5.2f}) | "
            f"V2={det['v2']:6.2f} "
            f"(err {v2_error:5.2f}) | "
            f"points={det['points']}"
        )

    if rows:
        mae_v1 = np.mean(
            [r["v1_error"] for r in rows]
        )

        valid_v2 = [
            r["v2_error"]
            for r in rows
            if r["v2_error"] is not None
        ]

        mae_v2 = (
            np.mean(valid_v2)
            if valid_v2
            else float("nan")
        )

        print()
        print("=" * 100)
        print(f"Matched objects: {len(rows)}")
        print(f"V1 MAE: {mae_v1:.3f} m")
        print(f"V2 MAE: {mae_v2:.3f} m")

    csv_path = OUTPUT_DIR / f"depth_validation_{FRAME}.csv"

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

        print("Saved:", csv_path)


if __name__ == "__main__":
    main()
