from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from adapters.falcon_adapter import run_falcon


FRAME = "000149"
QUERY = "car"

DEDUP_IOU = 0.50
MIN_LIDAR_POINTS = 3

KITTI_ROOT = (
    Path.home()
    / "Major_Project/datasets/KITTI/training"
)

OUTPUT_ROOT = (
    Path.home()
    / "Major_Project/model_outputs/falcon_lidar_fusion"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_calibration(path):
    calib = {}

    with open(path, "r") as f:
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
        [xyz, np.ones((xyz.shape[0], 1))]
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
        & (depth > 0)
    )

    return (
        u[valid],
        v[valid],
        depth[valid],
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

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def deduplicate_boxes(boxes, threshold):
    if not boxes:
        return []

    n = len(boxes)

    adjacency = [
        {i}
        for i in range(n)
    ]

    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        stack = [start]
        cluster = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    stack.append(neighbour)

        clusters.append(cluster)

    deduplicated = []

    for cluster in clusters:
        cluster_boxes = np.array(
            [boxes[i] for i in cluster],
            dtype=float,
        )

        deduplicated.append(
            np.median(
                cluster_boxes,
                axis=0,
            )
        )

    return deduplicated


def main():
    image_path = (
        KITTI_ROOT
        / "image_2"
        / f"{FRAME}.png"
    )

    lidar_path = (
        KITTI_ROOT
        / "velodyne"
        / f"{FRAME}.bin"
    )

    calib_path = (
        KITTI_ROOT
        / "calib"
        / f"{FRAME}.txt"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_array = np.array(image)

    width, height = image.size

    points = np.fromfile(
        lidar_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    calib = load_calibration(
        calib_path
    )

    u, v, depth = project_lidar(
        points,
        calib,
        width,
        height,
    )

    result = run_falcon(
        image_path=image_path,
        query=QUERY,
        dtype="float16",
    )

    raw_boxes = []

    for detection in result["detections"]:
        box = detection["pixel_bbox"]

        raw_boxes.append(
            np.array(
                [
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                ],
                dtype=float,
            )
        )

    boxes = deduplicate_boxes(
        raw_boxes,
        DEDUP_IOU,
    )

    print()
    print("Raw Falcon boxes:", len(raw_boxes))
    print("Deduplicated boxes:", len(boxes))

    print()
    print("FALCON + LIDAR FUSION")
    print("-" * 70)

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.imshow(image_array)

    fused_count = 0

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        inside = (
            (u >= x1)
            & (u <= x2)
            & (v >= y1)
            & (v <= y2)
        )

        object_depths = depth[inside]

        rectangle = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        if len(object_depths) < MIN_LIDAR_POINTS:
            print(
                f"[{index:02d}] "
                f"insufficient LiDAR | "
                f"points={len(object_depths)}"
            )

            continue

        estimated_depth = float(
            np.median(object_depths)
        )

        fused_count += 1

        ax.scatter(
            u[inside],
            v[inside],
            c=object_depths,
            s=8,
            cmap="turbo",
            vmin=0,
            vmax=70,
        )

        ax.text(
            x1,
            max(0, y1 - 5),
            (
                f"car | "
                f"{estimated_depth:.1f} m | "
                f"{len(object_depths)} pts"
            ),
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
            },
        )

        print(
            f"[{index:02d}] "
            f"depth={estimated_depth:.2f}m | "
            f"LiDAR points={len(object_depths)}"
        )

    ax.set_title(
        f"Falcon Perception + LiDAR Fusion — KITTI {FRAME}"
    )

    ax.axis("off")
    plt.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / f"falcon_lidar_{FRAME}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Fused detections:", fused_count)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
