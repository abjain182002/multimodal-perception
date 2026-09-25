"""CPU reference 3D IoU for CLOCs candidate target preparation.

All boxes use rectified KITTI camera coordinates and CLOCs order:
    [x, y, z, length, height, width, rotation_y].
The y coordinate is the bottom of the box; camera y increases downwards.
This implements the camera-space intersection and vertical overlap used by
SECOND/CLOCs, with a rotated x-z rectangle for the bird's-eye footprint.
"""

import numpy as np


def _camera_boxes(boxes, name):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"{name} must have shape [N, 7]")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(boxes[:, 3:6] <= 0):
        raise ValueError(f"{name} dimensions must be positive")
    return boxes


def _bev_corners(boxes):
    # Clockwise corners in (x, z), matching the SECOND rotated-IoU kernel.
    unit = np.array([[-0.5, -0.5], [-0.5, 0.5],
                     [0.5, 0.5], [0.5, -0.5]])
    local = unit[None, :, :] * boxes[:, None, [3, 5]]
    cosine = np.cos(boxes[:, 6])[:, None]
    sine = np.sin(boxes[:, 6])[:, None]
    x = local[:, :, 0] * cosine + local[:, :, 1] * sine + boxes[:, None, 0]
    z = -local[:, :, 0] * sine + local[:, :, 1] * cosine + boxes[:, None, 2]
    return np.stack((x, z), axis=-1)


def _cross(edge_start, edge_end, point):
    direction = edge_end - edge_start
    offset = point - edge_start
    return direction[0] * offset[1] - direction[1] * offset[0]


def _intersection_area(subject, clip):
    """Clip a clockwise convex quadrilateral against another one."""
    polygon = list(subject)
    for edge_start, edge_end in zip(clip, np.roll(clip, -1, axis=0)):
        if not polygon:
            return 0.0
        output = []
        previous = polygon[-1]
        previous_side = _cross(edge_start, edge_end, previous)
        for current in polygon:
            current_side = _cross(edge_start, edge_end, current)
            previous_inside = previous_side <= 0.0
            current_inside = current_side <= 0.0
            if current_inside != previous_inside:
                t = previous_side / (previous_side - current_side)
                output.append(previous + t * (current - previous))
            if current_inside:
                output.append(current)
            previous, previous_side = current, current_side
        polygon = output
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon)
    return 0.5 * abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                             - points[:, 1] * np.roll(points[:, 0], -1)))


def best_camera_3d_iou(candidate_boxes, gt_boxes):
    """Return the maximum rotated 3D IoU per candidate, or zero without GT.

    Candidate and GT boxes MUST already be in the same rectified camera
    frame. An AABB and height prefilter avoids polygon clipping for distant
    dense candidates. This function does not implement KITTI AP ignore rules.
    """
    candidates = _camera_boxes(candidate_boxes, "candidate_boxes")
    ground_truth = _camera_boxes(gt_boxes, "gt_boxes")
    best = np.zeros(candidates.shape[0], dtype=np.float32)
    if not candidates.size or not ground_truth.size:
        return best

    candidate_corners = _bev_corners(candidates)
    gt_corners = _bev_corners(ground_truth)
    candidate_min = candidate_corners.min(axis=1)
    candidate_max = candidate_corners.max(axis=1)
    candidate_volume = np.prod(candidates[:, 3:6], axis=1)
    gt_volume = np.prod(ground_truth[:, 3:6], axis=1)

    for gt_index, gt in enumerate(ground_truth):
        height = np.minimum(candidates[:, 1], gt[1]) - np.maximum(
            candidates[:, 1] - candidates[:, 4], gt[1] - gt[4]
        )
        possible = (
            (height > 0)
            & np.all(candidate_max > gt_corners[gt_index].min(axis=0), axis=1)
            & np.all(candidate_min < gt_corners[gt_index].max(axis=0), axis=1)
        )
        for candidate_index in np.flatnonzero(possible):
            area = _intersection_area(candidate_corners[candidate_index],
                                      gt_corners[gt_index])
            intersection = area * height[candidate_index]
            if intersection > 0:
                union = (candidate_volume[candidate_index]
                         + gt_volume[gt_index] - intersection)
                best[candidate_index] = max(
                    best[candidate_index], min(1.0, intersection / union)
                )
    return best
