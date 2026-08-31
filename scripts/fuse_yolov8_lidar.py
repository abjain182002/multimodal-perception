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

        # Median is more robust than mean to background points.
        estimated_depth = float(
            np.median(object_depths)
        )

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
            u[inside],
            v[inside],
            c=object_depths,
            s=8,
            cmap="turbo",
            vmin=0,
            vmax=70,
        )

        label = (
            f"{relevant_classes[cls_id]} "
            f"{confidence:.2f} | "
            f"{estimated_depth:.1f} m | "
            f"{len(object_depths)} pts"
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
            f"lidar_points={len(object_depths)}",
        )

    ax.set_title(
        f"YOLOv8x + LiDAR Fusion — KITTI {FRAME}"
    )

    ax.axis("off")
    plt.tight_layout()

    output = OUTPUT_ROOT / f"fusion_{FRAME}.png"

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
