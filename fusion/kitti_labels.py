#!/usr/bin/env python3

from pathlib import Path

import numpy as np


KITTI_GT_FIELD_COUNT = 15
KITTI_RESULT_FIELD_COUNT = 16


def parse_kitti_label_line(
    line,
):
    """
    Parse one KITTI object label/result line.

    Raw KITTI ground-truth order:
        type
        truncation
        occlusion
        alpha
        bbox_left bbox_top bbox_right bbox_bottom
        height width length
        x y z
        rotation_y

    An optional 16th field is interpreted as detection score.

    Returns
    -------
    dict
        Includes both the raw KITTI h/w/l dimensions and the
        CLOCs-compatible camera box:

            [x, y, z, length, height, width, rotation_y]
    """

    line = str(
        line
    ).strip()

    if not line:
        raise ValueError(
            "KITTI label line cannot be empty"
        )

    fields = line.split()

    if len(fields) not in (
        KITTI_GT_FIELD_COUNT,
        KITTI_RESULT_FIELD_COUNT,
    ):
        raise ValueError(
            "KITTI label line must contain 15 fields "
            "or 16 fields when a score is present"
        )

    name = fields[0]

    try:
        truncation = float(
            fields[1]
        )

        occlusion = int(
            fields[2]
        )

        alpha = float(
            fields[3]
        )

        bbox = np.asarray(
            [
                float(value)
                for value in fields[4:8]
            ],
            dtype=np.float32,
        )

        dimensions_hwl = np.asarray(
            [
                float(value)
                for value in fields[8:11]
            ],
            dtype=np.float32,
        )

        location = np.asarray(
            [
                float(value)
                for value in fields[11:14]
            ],
            dtype=np.float32,
        )

        rotation_y = float(
            fields[14]
        )

        score = (
            float(
                fields[15]
            )
            if len(fields)
            == KITTI_RESULT_FIELD_COUNT
            else None
        )

    except ValueError as exc:
        raise ValueError(
            "KITTI label contains an invalid numeric field"
        ) from exc

    numeric_values = np.concatenate(
        [
            np.asarray(
                [
                    truncation,
                    alpha,
                    rotation_y,
                ],
                dtype=np.float32,
            ),
            bbox,
            dimensions_hwl,
            location,
        ]
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "KITTI label contains non-finite numeric values"
        )

    if (
        score is not None
        and not np.isfinite(
            score
        )
    ):
        raise ValueError(
            "KITTI score must be finite"
        )

    height = float(
        dimensions_hwl[0]
    )

    width = float(
        dimensions_hwl[1]
    )

    length = float(
        dimensions_hwl[2]
    )

    x = float(
        location[0]
    )

    y = float(
        location[1]
    )

    z = float(
        location[2]
    )

    camera_box_lhw = np.asarray(
        [
            x,
            y,
            z,
            length,
            height,
            width,
            rotation_y,
        ],
        dtype=np.float32,
    )

    return {
        "name":
            name,

        "truncation":
            truncation,

        "occlusion":
            occlusion,

        "alpha":
            alpha,

        "bbox":
            bbox,

        "dimensions_hwl":
            dimensions_hwl,

        "location":
            location,

        "rotation_y":
            rotation_y,

        "score":
            score,

        "camera_box_lhw":
            camera_box_lhw,
    }


def parse_kitti_label_file(
    path,
):
    """
    Parse all non-empty lines from one KITTI label file.
    """

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"KITTI label file not found: {path}"
        )

    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = parse_kitti_label_line(
                line
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid KITTI label at "
                f"{path}:{line_number}: {exc}"
            ) from exc

        records.append(
            record
        )

    return records


def extract_class_camera_boxes(
    records,
    target_class="Car",
):
    """
    Extract camera-coordinate 3D boxes for one exact KITTI class.

    Output format:
        [x, y, z, length, height, width, rotation_y]

    Shape:
        [N, 7]
    """

    target_class = str(
        target_class
    ).strip()

    if not target_class:
        raise ValueError(
            "target_class cannot be empty"
        )

    selected = [
        record[
            "camera_box_lhw"
        ]
        for record in records
        if record[
            "name"
        ] == target_class
    ]

    if not selected:
        return np.zeros(
            (
                0,
                7,
            ),
            dtype=np.float32,
        )

    boxes = np.stack(
        selected,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return boxes


POINTPILLARS_KITTI_POINT_CLOUD_RANGE = np.asarray(
    [
        0.0,
        -39.68,
        -3.0,
        69.12,
        39.68,
        1.0,
    ],
    dtype=np.float32,
)


def camera_boxes_to_clocs_lidar_boxes(
    camera_boxes_lhw,
    lidar_to_camera,
):
    """
    Convert CLOCs-format camera boxes to the LiDAR representation
    used by the pinned CLOCs/SECOND preprocessing path.

    Input camera format:
        [x, y, z, length, height, width, rotation_y]

    Output LiDAR format:
        [x, y, z, width, length, height, rotation]

    The box centre is transformed by the inverse rectified
    LiDAR-to-camera matrix. The rotation value is preserved to
    match the pinned CLOCs/SECOND preprocessing convention.
    """

    camera_boxes = np.asarray(
        camera_boxes_lhw,
        dtype=np.float32,
    )

    if (
        camera_boxes.ndim != 2
        or camera_boxes.shape[1] != 7
    ):
        raise ValueError(
            "camera_boxes_lhw must have shape [N, 7]"
        )

    if not np.isfinite(
        camera_boxes
    ).all():
        raise ValueError(
            "camera_boxes_lhw contains non-finite values"
        )

    lidar_to_camera = np.asarray(
        lidar_to_camera,
        dtype=np.float64,
    )

    if lidar_to_camera.shape != (4, 4):
        raise ValueError(
            "lidar_to_camera must have shape [4, 4]"
        )

    if not np.isfinite(
        lidar_to_camera
    ).all():
        raise ValueError(
            "lidar_to_camera contains non-finite values"
        )

    if camera_boxes.shape[0] == 0:
        return np.zeros(
            (0, 7),
            dtype=np.float32,
        )

    camera_to_lidar = np.linalg.inv(
        lidar_to_camera
    )

    xyz_camera = camera_boxes[
        :,
        :3,
    ].astype(
        np.float64
    )

    xyz_camera_h = np.concatenate(
        [
            xyz_camera,
            np.ones(
                (
                    xyz_camera.shape[0],
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    xyz_lidar_h = (
        xyz_camera_h
        @ camera_to_lidar.T
    )

    xyz_lidar = xyz_lidar_h[
        :,
        :3,
    ]

    length = camera_boxes[
        :,
        3:4,
    ]

    height = camera_boxes[
        :,
        4:5,
    ]

    width = camera_boxes[
        :,
        5:6,
    ]

    rotation = camera_boxes[
        :,
        6:7,
    ]

    boxes_lidar = np.concatenate(
        [
            xyz_lidar.astype(
                np.float32
            ),
            width,
            length,
            height,
            rotation,
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    return boxes_lidar


def lidar_bev_box_corners(
    boxes_lidar,
):
    """
    Compute rotated BEV corners using the same clockwise-positive
    convention as the pinned SECOND/CLOCs center_to_corner_box2d.

    Input:
        [x, y, z, width, length, height, rotation]

    Output:
        [N, 4, 2]
    """

    boxes = np.asarray(
        boxes_lidar,
        dtype=np.float32,
    )

    if (
        boxes.ndim != 2
        or boxes.shape[1] != 7
    ):
        raise ValueError(
            "boxes_lidar must have shape [N, 7]"
        )

    if not np.isfinite(
        boxes
    ).all():
        raise ValueError(
            "boxes_lidar contains non-finite values"
        )

    if boxes.shape[0] == 0:
        return np.zeros(
            (
                0,
                4,
                2,
            ),
            dtype=np.float32,
        )

    base = np.asarray(
        [
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )

    dimensions = boxes[
        :,
        3:5,
    ]

    local = (
        base[
            None,
            :,
            :,
        ]
        * dimensions[
            :,
            None,
            :,
        ]
    )

    angles = boxes[
        :,
        6,
    ]

    cosine = np.cos(
        angles
    )

    sine = np.sin(
        angles
    )

    rotated_x = (
        local[
            :,
            :,
            0,
        ]
        * cosine[
            :,
            None,
        ]
        + local[
            :,
            :,
            1,
        ]
        * sine[
            :,
            None,
        ]
    )

    rotated_y = (
        -local[
            :,
            :,
            0,
        ]
        * sine[
            :,
            None,
        ]
        + local[
            :,
            :,
            1,
        ]
        * cosine[
            :,
            None,
        ]
    )

    corners = np.stack(
        [
            rotated_x,
            rotated_y,
        ],
        axis=-1,
    )

    corners += boxes[
        :,
        None,
        0:2,
    ]

    return corners.astype(
        np.float32,
        copy=False,
    )


def filter_lidar_boxes_by_bev_range(
    boxes_lidar,
    point_cloud_range,
):
    """
    Reproduce the CLOCs/SECOND GT BEV range-filtering criterion:
    retain a box when at least one of its four rotated BEV corners
    lies inside the detector XY range.
    """

    boxes = np.asarray(
        boxes_lidar,
        dtype=np.float32,
    )

    point_cloud_range = np.asarray(
        point_cloud_range,
        dtype=np.float32,
    ).reshape(-1)

    if point_cloud_range.shape != (6,):
        raise ValueError(
            "point_cloud_range must have six values"
        )

    if not np.isfinite(
        point_cloud_range
    ).all():
        raise ValueError(
            "point_cloud_range contains non-finite values"
        )

    x_min = float(
        point_cloud_range[0]
    )

    y_min = float(
        point_cloud_range[1]
    )

    x_max = float(
        point_cloud_range[3]
    )

    y_max = float(
        point_cloud_range[4]
    )

    if (
        x_min >= x_max
        or y_min >= y_max
    ):
        raise ValueError(
            "point_cloud_range XY bounds are invalid"
        )

    corners = lidar_bev_box_corners(
        boxes
    )

    inside = (
        (corners[:, :, 0] >= x_min)
        & (corners[:, :, 0] <= x_max)
        & (corners[:, :, 1] >= y_min)
        & (corners[:, :, 1] <= y_max)
    )

    return np.any(
        inside,
        axis=1,
    )


def prepare_clocs_car_ground_truth(
    records,
    lidar_to_camera,
    point_cloud_range=POINTPILLARS_KITTI_POINT_CLOUD_RANGE,
):
    """
    Prepare Car GT using the CLOCs training-range semantics.

    Returns both the filtered camera boxes used for later 3D-IoU
    target construction and their corresponding LiDAR boxes used
    for detector-range filtering.
    """

    camera_boxes = extract_class_camera_boxes(
        records,
        target_class="Car",
    )

    lidar_boxes = camera_boxes_to_clocs_lidar_boxes(
        camera_boxes,
        lidar_to_camera,
    )

    keep_mask = filter_lidar_boxes_by_bev_range(
        lidar_boxes,
        point_cloud_range,
    )

    return {
        "camera_boxes":
            camera_boxes[
                keep_mask
            ],

        "lidar_boxes":
            lidar_boxes[
                keep_mask
            ],

        "keep_mask":
            keep_mask,

        "num_car_before_range_filter":
            int(
                camera_boxes.shape[0]
            ),

        "num_car_after_range_filter":
            int(
                keep_mask.sum()
            ),
    }
