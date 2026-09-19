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


# ---------------------------------------------------------
# CONTROLLED EXPERIMENT SETTINGS
# ---------------------------------------------------------

QUERY = "car"

DEDUP_IOU = 0.50
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
    / "Major_Project/model_outputs/falcon_lidar_multiframe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# FALCON OUTPUT UTILITIES
# ---------------------------------------------------------

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


def bbox_to_pixels(
    bbox,
    width,
    height,
):
    cx = float(bbox["x"])
    cy = float(bbox["y"])
    bw = float(bbox["w"])
    bh = float(bbox["h"])

    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height

    return np.array(
        [
            np.clip(x1, 0, width),
            np.clip(y1, 0, height),
            np.clip(x2, 0, width),
            np.clip(y2, 0, height),
        ],
        dtype=float,
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


def deduplicate_boxes(
    boxes,
    threshold,
):
    """
    Score-free connected-component IoU clustering.

    Falcon does not provide a conventional confidence
    score, so overlapping predictions are clustered and
    represented by the coordinate-wise median box.
    """

    if not boxes:
        return []

    n = len(boxes)

    adjacency = [
        {i}
        for i in range(n)
    ]

    for i in range(n):
        for j in range(i + 1, n):
            if (
                iou(
                    boxes[i],
                    boxes[j],
                )
                >= threshold
            ):
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    stack.append(neighbour)

        clusters.append(component)

    representatives = []

    for cluster in clusters:
        cluster_boxes = np.array(
            [
                boxes[i]
                for i in cluster
            ],
            dtype=float,
        )

        representative = np.median(
            cluster_boxes,
            axis=0,
        )

        representatives.append(
            representative
        )

    return representatives


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
        "Falcon + LiDAR "
        "multi-frame holdout evaluation"
    )

    print(
        "Development frames:"
    )

    print(
        ", ".join(
            HOLDOUT_FRAMES
        )
    )

    print()

    print(
        "Loading Falcon Perception MLX model..."
    )

    load_start = (
        time.perf_counter()
    )

    model, tokenizer, model_args = (
        load_and_prepare_model(
            hf_model_id=PERCEPTION_MODEL_ID,
            dtype="float16",
            backend="mlx",
        )
    )

    engine = (
        BatchInferenceEngine(
            model,
            tokenizer,
        )
    )

    model_load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Model loaded in "
        f"{model_load_time:.2f}s"
    )

    prompt = build_prompt_for_task(
        QUERY,
        "detection",
    )

    all_rows = []
    generation_times = []

    total_gt = 0
    total_raw = 0
    total_dedup = 0
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
        # Falcon inference
        # -------------------------

        batch = (
            process_batch_and_generate(
                tokenizer,
                [(image, prompt)],
                max_length=(
                    model_args.max_seq_len
                ),
                min_dimension=256,
                max_dimension=1024,
            )
        )

        start = (
            time.perf_counter()
        )

        _, aux_outputs = (
            engine.generate(
                tokens=batch[
                    "tokens"
                ],
                pos_t=batch[
                    "pos_t"
                ],
                pos_hw=batch[
                    "pos_hw"
                ],
                pixel_values=batch[
                    "pixel_values"
                ],
                pixel_mask=batch[
                    "pixel_mask"
                ],
                max_new_tokens=200,
                temperature=0.0,
                task="detection",
            )
        )

        generation_time = (
            time.perf_counter()
            - start
        )

        generation_times.append(
            generation_time
        )

        raw_boxes = [
            bbox_to_pixels(
                bbox,
                width,
                height,
            )
            for bbox in pair_bbox_entries(
                aux_outputs[
                    0
                ].bboxes_raw
            )
        ]

        boxes = (
            deduplicate_boxes(
                raw_boxes,
                DEDUP_IOU,
            )
        )

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

        total_dedup += len(
            boxes
        )

        total_fused += len(
            fused_detections
        )

        print(
            f"[{index:02d}/{len(HOLDOUT_FRAMES):02d}] "
            f"frame={frame} | "
            f"GT={len(ground_truth):2d} | "
            f"raw={len(raw_boxes):3d} | "
            f"dedup={len(boxes):2d} | "
            f"fused={len(fused_detections):2d} | "
            f"matches={len(matches):2d} | "
            f"time={generation_time:.2f}s"
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

    mean_generation = float(
        np.mean(
            generation_times
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
        f"Raw Falcon boxes:          "
        f"{total_raw}"
    )

    print(
        f"Deduplicated boxes:        "
        f"{total_dedup}"
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
        f"Mean Falcon generation:    "
        f"{mean_generation:.3f} s/frame"
    )

    print(
        f"Approx generation FPS:     "
        f"{1.0 / mean_generation:.3f}"
    )

    # -----------------------------------------------------
    # SAVE MATCH-LEVEL RESULTS
    # -----------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "falcon_lidar_holdout_10.csv"
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