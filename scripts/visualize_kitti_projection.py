from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

FRAME = "000149"

KITTI_ROOT = Path.home() / "Major_Project/datasets/KITTI/training"
OUTPUT_ROOT = Path.home() / "Major_Project/model_outputs/kitti_projection"
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


def project_lidar_to_image(points, calib, width, height):
    P2 = calib["P2"].reshape(3, 4)
    R0 = calib["R0_rect"].reshape(3, 3)
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_4x4 = np.eye(4)
    R0_4x4[:3, :3] = R0

    Tr_4x4 = np.eye(4)
    Tr_4x4[:3, :] = Tr

    xyz = points[:, :3]

    xyz_hom = np.hstack(
        [xyz, np.ones((xyz.shape[0], 1))]
    )

    camera_points = (
        R0_4x4
        @ Tr_4x4
        @ xyz_hom.T
    ).T

    front_mask = camera_points[:, 2] > 0
    camera_points = camera_points[front_mask]

    depth = camera_points[:, 2]

    projected = (
        P2 @ camera_points.T
    ).T

    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    image_mask = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & (depth > 0)
    )

    return (
        u[image_mask],
        v[image_mask],
        depth[image_mask],
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

    u, v, depth = project_lidar_to_image(
        points,
        calib,
        width,
        height,
    )

    print("Frame:", FRAME)
    print("Image shape:", image.shape)
    print("Original LiDAR points:", len(points))
    print("Projected image points:", len(u))

    plt.figure(figsize=(15, 6))
    plt.imshow(image)

    scatter = plt.scatter(
        u,
        v,
        c=depth,
        s=2,
        cmap="turbo",
        vmin=0,
        vmax=70,
    )

    plt.colorbar(
        scatter,
        label="Depth from camera (m)",
    )

    plt.title(
        f"KITTI Camera–LiDAR Projection — Frame {FRAME}"
    )

    plt.axis("off")
    plt.tight_layout()

    output = OUTPUT_ROOT / f"lidar_on_camera_{FRAME}.png"

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
