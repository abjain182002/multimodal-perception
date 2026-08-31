from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


FRAME = "000149"

KITTI_ROOT = Path.home() / "Major_Project/datasets/KITTI/training"
OUTPUT_ROOT = Path.home() / "Major_Project/model_outputs/yolov8_lidar_fusion"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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
    lidar_points = points[front]

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
        lidar_points[valid],
    )


def main():
    image_path = KITTI_ROOT / "image_2" / f"{FRAME}.png"
    lidar_path = KITTI_ROOT / "velodyne" / f"{FRAME}.bin"
    calib_path = KITTI_ROOT / "calib" / f"{FRAME}.txt"

    image = np.array(Image.open(image_path))
    height, width = image.shape[:2]

    points = np.fromfile(
        lidar_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    calib = load_calibration(calib_path)

    u, v, depth, lidar = project_lidar(
        points,
        calib,
        width,
        height,
    )

    model = YOLO("yolov8x.pt")

    results = model.predict(
        source=str(image_path),
        device="mps",
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.imshow(image)

    detections_used = 0

    relevant_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        5: "bus",
        7: "truck",
    }

    for box in result.boxes:
        cls_id = int(box.cls.item())

        if cls_id not in relevant_classes:
            continue

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy()
        )

        inside = (
            (u >= x1)
            & (u <= x2)
            & (v >= y1)
            & (v <= y2)
        )

        object_depths = depth[inside]

        # Require several LiDAR returns for a meaningful estimate.
        if len(object_depths) < 3:
            continue

        # -------------------------------------------------
        # Foreground depth filtering
        # -------------------------------------------------
        # Sort depths and split when there is a large gap.
        # This helps separate the detected object from
        # background surfaces inside the same 2D box.
        sorted_depths = np.sort(object_depths)

        depth_gaps = np.diff(sorted_depths)
        split_indices = np.where(depth_gaps > 1.5)[0] + 1

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
            continue

        # Use the nearest substantial depth cluster
        # as the likely foreground object.
        foreground_cluster = min(
            valid_clusters,
            key=lambda cluster: np.median(cluster),
        )

        foreground_min = float(foreground_cluster.min())
        foreground_max = float(foreground_cluster.max())

        foreground_mask = (
            inside
            & (depth >= foreground_min)
            & (depth <= foreground_max)
        )

        foreground_depths = depth[foreground_mask]

        if len(foreground_depths) < 3:
            continue

        estimated_depth = float(
            np.median(foreground_depths)
        )

        foreground_count = len(foreground_depths)

        detections_used += 1

        rectangle = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        ax.scatter(
            u[foreground_mask],
            v[foreground_mask],
            c=foreground_depths,
            s=8,
            cmap="turbo",
            vmin=0,
            vmax=70,
        )

        label = (
            f"{relevant_classes[cls_id]} "
            f"{confidence:.2f} | "
            f"{estimated_depth:.1f} m | "
            f"{foreground_count}/{len(object_depths)} pts"
        )

        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
            },
        )

        print(
            relevant_classes[cls_id],
            f"confidence={confidence:.3f}",
            f"depth={estimated_depth:.2f}m",
            f"foreground_points={foreground_count}",
            f"total_box_points={len(object_depths)}",
        )

    ax.set_title(
        f"YOLOv8x + LiDAR Fusion V2 — KITTI {FRAME}"
    )

    ax.axis("off")
    plt.tight_layout()

    output = OUTPUT_ROOT / f"fusion_v2_{FRAME}.png"

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print("")
    print("Fused detections:", detections_used)
    print("Saved:", output)


if __name__ == "__main__":
    main()
