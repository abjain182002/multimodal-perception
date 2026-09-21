from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image

from ultralytics import YOLO


# ---------------------------------------------------------
# CONTROLLED EXPERIMENT SETTINGS
# ---------------------------------------------------------

YOLO_CONF = 0.25
YOLO_DEVICE = "mps"
MATCH_IOU = 0.30
MIN_LIDAR_POINTS = 3

# Same development frames used in the previous
# Falcon and YOLO controlled camera comparison.
HOLDOUT_FRAMES = [
    "000642",
    "000666",
    "000704",
    "001506",
    "003236",
    "003280",
    "004891",
    "005215",
    "005783",
    "006419",
]

KITTI_ROOT = (
    Path.home()
    / "Major_Project/datasets/KITTI/training"
)

OUTPUT_DIR = (
    Path.home()
    / "Major_Project/model_outputs/yolo26_lidar_multiframe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# BOUNDING-BOX UTILITIES
# ---------------------------------------------------------

def iou(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------
# KITTI CALIBRATION + LIDAR
# ---------------------------------------------------------

def load_calibration(path):
    calib = {}

    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = (
                line.strip().split(
                    ":",
                    1,
                )
            )

            if value.strip():
                calib[key] = np.array(
                    [
                        float(x)
                        for x
                        in value.split()
                    ],
                    dtype=np.float64,
                )

    return calib


def project_lidar(
    points,
    calib,
    width,
    height,
):
    P2 = calib["P2"].reshape(
        3,
        4,
    )

    R0 = calib[
        "R0_rect"
    ].reshape(
        3,
        3,
    )

    Tr = calib[
        "Tr_velo_to_cam"
    ].reshape(
        3,
        4,
    )

    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4)
    Tr_4[:3, :] = Tr

    xyz_hom = np.hstack(
        [
            points[:, :3],
            np.ones(
                (
                    len(points),
                    1,
                )
            ),
        ]
    )

    camera_points = (
        R0_4
        @ Tr_4
        @ xyz_hom.T
    ).T

    front = (
        camera_points[:, 2]
        > 0
    )

    camera_points = (
        camera_points[front]
    )

    projected = (
        P2
        @ camera_points.T
    ).T

    u = (
        projected[:, 0]
        / projected[:, 2]
    )

    v = (
        projected[:, 1]
        / projected[:, 2]
    )

    depth = (
        camera_points[:, 2]
    )

    valid = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & (depth > 0)
    )

    return (
        u[valid],
        v[valid],
        depth[valid],
    )


# ---------------------------------------------------------
# KITTI GROUND TRUTH
# ---------------------------------------------------------

def load_ground_truth(frame):
    label_path = (
        KITTI_ROOT
        / "label_2"
        / f"{frame}.txt"
    )

    ground_truth = []

    with open(label_path) as f:
        for line in f:
            parts = (
                line.strip().split()
            )

            if parts[0] != "Car":
                continue

            ground_truth.append(
                {
                    "bbox": np.array(
                        list(
                            map(
                                float,
                                parts[4:8],
                            )
                        ),
                        dtype=float,
                    ),
                    # KITTI camera-coordinate
                    # object-centre z location.
                    "depth": float(
                        parts[13]
                    ),
                    "truncation": float(
                        parts[1]
                    ),
                    "occlusion": int(
                        parts[2]
                    ),
                }
            )

    return ground_truth


# ---------------------------------------------------------
# MATCHING
# ---------------------------------------------------------

def match_detections_to_gt(
    detections,
    ground_truth,
    threshold,
):
    candidates = []

    for det_idx, detection in enumerate(
        detections
    ):
        for gt_idx, truth in enumerate(
            ground_truth
        ):
            score = iou(
                detection["bbox"],
                truth["bbox"],
            )

            candidates.append(
                (
                    score,
                    det_idx,
                    gt_idx,
                )
            )

    candidates.sort(
        reverse=True
    )

    used_detections = set()
    used_gt = set()

    matches = []

    for (
        score,
        det_idx,
        gt_idx,
    ) in candidates:

        if score < threshold:
            continue

        if (
            det_idx
            in used_detections
        ):
            continue

        if gt_idx in used_gt:
            continue

        used_detections.add(
            det_idx
        )

        used_gt.add(
            gt_idx
        )

        matches.append(
            (
                score,
                det_idx,
                gt_idx,
            )
        )

    return matches


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main():

    print(
        "YOLO26x + LiDAR "
        "multi-frame holdout evaluation"
    )

    print(
        "Holdout frames:"
    )

    print(
        ", ".join(
            HOLDOUT_FRAMES
        )
    )

    print()

    print(
        "Loading YOLO26x..."
    )

    load_start = (
        time.perf_counter()
    )

    model = YOLO(
        "yolo26x.pt"
    )

    model_load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Model loaded in "
        f"{model_load_time:.2f}s"
    )

    all_rows = []
    inference_times = []

    total_gt = 0
    total_raw = 0
    total_fused = 0

    print()
    print("=" * 92)

    for index, frame in enumerate(
        HOLDOUT_FRAMES,
        start=1,
    ):

        image_path = (
            KITTI_ROOT
            / "image_2"
            / f"{frame}.png"
        )

        lidar_path = (
            KITTI_ROOT
            / "velodyne"
            / f"{frame}.bin"
        )

        calib_path = (
            KITTI_ROOT
            / "calib"
            / f"{frame}.txt"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = (
            image.size
        )

        # -------------------------
        # YOLO26x inference
        # -------------------------

        start = (
            time.perf_counter()
        )

        result = model.predict(
            source=str(image_path),
            device=YOLO_DEVICE,
            conf=YOLO_CONF,
            verbose=False,
        )[0]

        inference_time = (
            time.perf_counter()
            - start
        )

        inference_times.append(
            inference_time
        )

        raw_boxes = []

        for detection in result.boxes:

            cls_id = int(
                detection.cls.item()
            )

            if (
                model.names[cls_id]
                != "car"
            ):
                continue

            raw_boxes.append(
                detection.xyxy[
                    0
                ].cpu().numpy()
            )

        # YOLO already performs its own confidence/NMS
        # processing, so no Falcon-style deduplication
        # is applied here.
        boxes = raw_boxes

        # -------------------------
        # LiDAR projection
        # -------------------------

        points = np.fromfile(
            lidar_path,
            dtype=np.float32,
        ).reshape(
            -1,
            4,
        )

        calib = (
            load_calibration(
                calib_path
            )
        )

        u, v, depth = (
            project_lidar(
                points,
                calib,
                width,
                height,
            )
        )

        # -------------------------
        # Frozen V1 fusion method
        # -------------------------

        fused_detections = []

        for box in boxes:

            x1, y1, x2, y2 = box

            inside = (
                (u >= x1)
                & (u <= x2)
                & (v >= y1)
                & (v <= y2)
            )

            object_depths = (
                depth[inside]
            )

            if (
                len(object_depths)
                < MIN_LIDAR_POINTS
            ):
                continue

            estimated_depth = float(
                np.median(
                    object_depths
                )
            )

            fused_detections.append(
                {
                    "bbox": box,
                    "estimated_depth": (
                        estimated_depth
                    ),
                    "lidar_points": (
                        len(
                            object_depths
                        )
                    ),
                }
            )

        # -------------------------
        # KITTI GT
        # -------------------------

        ground_truth = (
            load_ground_truth(
                frame
            )
        )

        matches = (
            match_detections_to_gt(
                fused_detections,
                ground_truth,
                MATCH_IOU,
            )
        )

        # -------------------------
        # Record matched depth data
        # -------------------------

        for (
            match_iou,
            det_idx,
            gt_idx,
        ) in matches:

            detection = (
                fused_detections[
                    det_idx
                ]
            )

            truth = (
                ground_truth[
                    gt_idx
                ]
            )

            signed_error = (
                detection[
                    "estimated_depth"
                ]
                - truth["depth"]
            )

            all_rows.append(
                {
                    "frame": frame,
                    "iou": match_iou,
                    "gt_depth": (
                        truth["depth"]
                    ),
                    "estimated_depth": (
                        detection[
                            "estimated_depth"
                        ]
                    ),
                    "signed_error": (
                        signed_error
                    ),
                    "absolute_error": (
                        abs(
                            signed_error
                        )
                    ),
                    "lidar_points": (
                        detection[
                            "lidar_points"
                        ]
                    ),
                    "truncation": (
                        truth[
                            "truncation"
                        ]
                    ),
                    "occlusion": (
                        truth[
                            "occlusion"
                        ]
                    ),
                }
            )

        total_gt += len(
            ground_truth
        )

        total_raw += len(
            raw_boxes
        )

        total_fused += len(
            fused_detections
        )

        print(
            f"[{index:02d}/{len(HOLDOUT_FRAMES):02d}] "
            f"frame={frame} | "
            f"GT={len(ground_truth):2d} | "
            f"YOLO={len(raw_boxes):3d} | "
            f"fused={len(fused_detections):2d} | "
            f"matches={len(matches):2d} | "
            f"time={inference_time:.3f}s"
        )

    print()
    print("=" * 92)

    if not all_rows:
        print(
            "No matched fused detections found."
        )
        return

    # -----------------------------------------------------
    # AGGREGATE DEPTH METRICS
    # -----------------------------------------------------

    signed_errors = np.array(
        [
            row["signed_error"]
            for row
            in all_rows
        ],
        dtype=float,
    )

    absolute_errors = np.abs(
        signed_errors
    )

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    median_ae = float(
        np.median(
            absolute_errors
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                signed_errors ** 2
            )
        )
    )

    mean_signed_error = float(
        np.mean(
            signed_errors
        )
    )

    mean_inference = float(
        np.mean(
            inference_times
        )
    )

    print(
        "AGGREGATE RESULTS"
    )

    print("-" * 92)

    print(
        f"Frames evaluated:          "
        f"{len(HOLDOUT_FRAMES)}"
    )

    print(
        f"Ground-truth cars:         "
        f"{total_gt}"
    )

    print(
        f"YOLO26x car detections:    "
        f"{total_raw}"
    )

    print(
        f"LiDAR-fused detections:    "
        f"{total_fused}"
    )

    print(
        f"Matched fused cars:        "
        f"{len(all_rows)}"
    )

    print()

    print(
        f"MAE:                       "
        f"{mae:.3f} m"
    )

    print(
        f"Median AE:                 "
        f"{median_ae:.3f} m"
    )

    print(
        f"RMSE:                      "
        f"{rmse:.3f} m"
    )

    print(
        f"Mean signed error:         "
        f"{mean_signed_error:.3f} m"
    )

    print()

    print(
        f"Mean YOLO inference:       "
        f"{mean_inference:.3f} s/frame"
    )

    print(
        f"Approx inference FPS:      "
        f"{1.0 / mean_inference:.3f}"
    )

    # -----------------------------------------------------
    # SAVE MATCH-LEVEL RESULTS
    # -----------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "yolo26_lidar_holdout_10.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=(
                    all_rows[
                        0
                    ].keys()
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    print()
    print(
        "Saved:",
        csv_path,
    )


if __name__ == "__main__":
    main()