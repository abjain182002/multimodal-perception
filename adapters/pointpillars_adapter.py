#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from fusion.clocs import DISTANCE_NORMALIZER


def load_kitti_calibration(path):
    """
    Load numeric matrices from a KITTI calibration text file.
    """

    path = Path(path)

    calibration = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.strip().split(
                ":",
                1,
            )

            value = value.strip()

            if not value:
                continue

            calibration[key] = np.array(
                [
                    float(item)
                    for item in value.split()
                ],
                dtype=np.float32,
            )

    required = {
        "P2",
        "R0_rect",
        "Tr_velo_to_cam",
    }

    missing = required.difference(
        calibration
    )

    if missing:
        raise ValueError(
            "Missing KITTI calibration fields: "
            + ", ".join(sorted(missing))
        )

    return calibration


def build_kitti_lidar2cam(calibration):
    """
    Build the KITTI rectified LiDAR-to-camera transform:

        lidar2cam = R0_rect @ Tr_velo_to_cam
    """

    r0_rect = np.eye(
        4,
        dtype=np.float32,
    )

    r0_rect[:3, :3] = calibration[
        "R0_rect"
    ].reshape(3, 3)

    tr_velo_to_cam = np.eye(
        4,
        dtype=np.float32,
    )

    tr_velo_to_cam[:3, :] = calibration[
        "Tr_velo_to_cam"
    ].reshape(3, 4)

    return (
        r0_rect
        @ tr_velo_to_cam
    ).astype(np.float32)


def load_pointpillars_model(
    config_path,
    checkpoint_path,
    device="cuda:0",
):
    """
    Load the pinned MMDetection3D PointPillars model.

    MMDetection3D is imported lazily so this module can still
    be imported and unit-tested on systems without MMDetection3D.
    """

    try:
        from mmdet3d.apis import init_model
    except ImportError as exc:
        raise ImportError(
            "MMDetection3D is required to load PointPillars."
        ) from exc

    return init_model(
        str(config_path),
        str(checkpoint_path),
        device=device,
    )


def extract_dense_pointpillars_candidates(
    model,
    pointcloud_path,
    calibration_path,
    image_width,
    image_height,
):
    """
    Extract the dense pre-NMS candidate set from PointPillars.

    The returned dictionary contains:

        boxes_lidar
        boxes_camera (rectified KITTI camera [x,y,z,l,h,w,ry])
        scores_3d
        direction_labels
        projected_boxes
        distance_82
        candidate_index

    This operates before the normal PointPillars nms_pre ranking,
    score threshold and NMS stages.
    """

    try:
        import torch

        from mmdet3d.apis import inference_detector
        from mmdet3d.structures import (
            Box3DMode,
            LiDARInstance3DBoxes,
            points_cam2img,
        )

    except ImportError as exc:
        raise ImportError(
            "PyTorch and MMDetection3D are required "
            "for dense PointPillars extraction."
        ) from exc

    pointcloud_path = Path(
        pointcloud_path
    )

    calibration = load_kitti_calibration(
        calibration_path
    )

    head_output = {}

    def capture_head_output(
        module,
        inputs,
        output,
    ):
        head_output["value"] = output

    hook = model.bbox_head.register_forward_hook(
        capture_head_output
    )

    try:
        with torch.no_grad():
            inference_detector(
                model,
                str(pointcloud_path),
            )
    finally:
        hook.remove()

    if "value" not in head_output:
        raise RuntimeError(
            "PointPillars bbox-head output was not captured."
        )

    output = head_output["value"]

    if (
        not isinstance(output, tuple)
        or len(output) != 3
    ):
        raise RuntimeError(
            "Unexpected PointPillars bbox-head output."
        )

    cls_scores, bbox_preds, dir_cls_preds = output

    if (
        len(cls_scores) != 1
        or len(bbox_preds) != 1
        or len(dir_cls_preds) != 1
    ):
        raise RuntimeError(
            "The validated KITTI PointPillars configuration "
            "is expected to have one feature level."
        )

    head = model.bbox_head

    feature_map_size = cls_scores[0].shape[-2:]

    priors = head.prior_generator.grid_anchors(
        [feature_map_size],
        device=cls_scores[0].device,
    )[0].reshape(
        -1,
        head.box_code_size,
    ).float()

    cls_logits = (
        cls_scores[0][0]
        .permute(1, 2, 0)
        .reshape(
            -1,
            head.num_classes,
        )
        .float()
    )

    bbox_deltas = (
        bbox_preds[0][0]
        .permute(1, 2, 0)
        .reshape(
            -1,
            head.box_code_size,
        )
        .float()
    )

    direction_logits = (
        dir_cls_preds[0][0]
        .permute(1, 2, 0)
        .reshape(
            -1,
            2,
        )
        .float()
    )

    boxes_lidar = head.bbox_coder.decode(
        priors,
        bbox_deltas,
    )

    scores_3d = torch.sigmoid(
        cls_logits
    )[:, 0]

    direction_labels = torch.argmax(
        direction_logits,
        dim=1,
    )

    distance_82 = (
        torch.linalg.vector_norm(
            boxes_lidar[:, :2],
            dim=1,
        )
        / DISTANCE_NORMALIZER
    )

    lidar2cam = torch.as_tensor(
        build_kitti_lidar2cam(
            calibration
        ),
        dtype=boxes_lidar.dtype,
        device=boxes_lidar.device,
    )

    p2 = torch.as_tensor(
        calibration["P2"].reshape(
            3,
            4,
        ),
        dtype=boxes_lidar.dtype,
        device=boxes_lidar.device,
    )

    lidar_boxes_3d = LiDARInstance3DBoxes(
        boxes_lidar,
        box_dim=7,
    )

    camera_boxes = lidar_boxes_3d.convert_to(
        Box3DMode.CAM,
        lidar2cam,
    )

    camera_corners = camera_boxes.corners

    projected_corners = points_cam2img(
        camera_corners,
        p2,
    )

    min_xy = projected_corners.min(
        dim=1
    ).values

    max_xy = projected_corners.max(
        dim=1
    ).values

    min_xy[:, 0].clamp_(
        0,
        float(image_width),
    )

    min_xy[:, 1].clamp_(
        0,
        float(image_height),
    )

    max_xy[:, 0].clamp_(
        0,
        float(image_width),
    )

    max_xy[:, 1].clamp_(
        0,
        float(image_height),
    )

    projected_boxes = torch.cat(
        [
            min_xy,
            max_xy,
        ],
        dim=1,
    )

    tensors = {
        "boxes_lidar": boxes_lidar,
        "boxes_camera": camera_boxes.tensor,
        "scores_3d": scores_3d,
        "direction_labels": direction_labels,
        "projected_boxes": projected_boxes,
        "distance_82": distance_82,
    }

    for name, tensor in tensors.items():
        if not torch.isfinite(
            tensor.float()
        ).all():
            raise RuntimeError(
                f"Non-finite values found in {name}."
            )

    num_candidates = boxes_lidar.shape[0]

    if not (
        camera_boxes.tensor.shape[0]
        ==
        scores_3d.shape[0]
        == direction_labels.shape[0]
        == projected_boxes.shape[0]
        == distance_82.shape[0]
        == num_candidates
    ):
        raise RuntimeError(
            "Dense PointPillars candidate arrays "
            "have inconsistent lengths."
        )

    return {
        "boxes_lidar": (
            boxes_lidar
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        "boxes_camera": (
            camera_boxes.tensor
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        "scores_3d": (
            scores_3d
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        "direction_labels": (
            direction_labels
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        ),

        "projected_boxes": (
            projected_boxes
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        "distance_82": (
            distance_82
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        "candidate_index": np.arange(
            num_candidates,
            dtype=np.int32,
        ),

        "feature_map_size": (
            int(feature_map_size[0]),
            int(feature_map_size[1]),
        ),

        "num_candidates": int(
            num_candidates
        ),
    }
