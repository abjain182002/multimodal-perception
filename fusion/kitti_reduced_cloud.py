"""Recreate pinned MMDetection3D KITTI camera-FOV reduced point clouds.

Matches tools/dataset_converters/kitti_converter.py at
fe25f7a51d36e3702f961e198894580d83c4387b, front_camera_id=2.
"""
from pathlib import Path

import numpy as np


def reduce_kitti_frame(raw_path, calibration, image_height, image_width, output_path):
    from mmdet3d.structures.ops import box_np_ops

    points = np.fromfile(str(raw_path), dtype=np.float32).reshape(-1, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = calibration['R0_rect'].reshape(3, 3)
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = calibration['Tr_velo_to_cam'].reshape(3, 4)
    projection = np.eye(4, dtype=np.float64)
    projection[:3, :] = calibration['P2'].reshape(3, 4)
    reduced = box_np_ops.remove_outside_points(
        points, rect, velo_to_cam, projection, (image_height, image_width))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reduced.astype(np.float32, copy=False).tofile(target)
    return len(points), len(reduced)
