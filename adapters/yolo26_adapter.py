#!/usr/bin/env python3

from pathlib import Path

import numpy as np


DEFAULT_CONFIDENCE = 0.25
DEFAULT_CLASS_NAME = "car"


def standardize_yolo_detections(
    boxes_xyxy,
    scores,
    class_ids,
    class_names,
    target_class=DEFAULT_CLASS_NAME,
):
    """
    Convert YOLO detections into the arrays used by CLOCs.

    The original detector ordering is preserved.
    No additional confidence filtering or NMS is applied here.
    """

    boxes_xyxy = np.asarray(
        boxes_xyxy,
        dtype=np.float32,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    ).reshape(-1)

    class_ids = np.asarray(
        class_ids,
        dtype=np.int64,
    ).reshape(-1)

    if boxes_xyxy.size == 0:
        boxes_xyxy = np.empty(
            (0, 4),
            dtype=np.float32,
        )

    if (
        boxes_xyxy.ndim != 2
        or boxes_xyxy.shape[1] != 4
    ):
        raise ValueError(
            "boxes_xyxy must have shape [N, 4]"
        )

    num_detections = boxes_xyxy.shape[0]

    if scores.shape[0] != num_detections:
        raise ValueError(
            "scores length must match boxes_xyxy"
        )

    if class_ids.shape[0] != num_detections:
        raise ValueError(
            "class_ids length must match boxes_xyxy"
        )

    keep = []

    for index, class_id in enumerate(
        class_ids
    ):
        class_id = int(class_id)

        if isinstance(
            class_names,
            dict,
        ):
            class_name = class_names[
                class_id
            ]
        else:
            class_name = class_names[
                class_id
            ]

        if (
            str(class_name).lower()
            == str(target_class).lower()
        ):
            keep.append(index)

    keep = np.asarray(
        keep,
        dtype=np.int64,
    )

    return {
        "boxes_2d": (
            boxes_xyxy[keep]
            .astype(
                np.float32,
                copy=False,
            )
        ),

        "scores_2d": (
            scores[keep]
            .astype(
                np.float32,
                copy=False,
            )
        ),

        "class_ids": (
            class_ids[keep]
            .astype(
                np.int64,
                copy=False,
            )
        ),

        "detection_index": (
            keep.astype(
                np.int32,
                copy=False,
            )
        ),

        "num_detections": int(
            len(keep)
        ),
    }


def load_yolo26_model(
    weights_path,
):
    """
    Load YOLO26 using Ultralytics.

    Ultralytics is imported lazily so the adapter's pure
    standardization logic can be unit-tested independently.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required to load YOLO26."
        ) from exc

    return YOLO(
        str(weights_path)
    )


def extract_yolo26_detections(
    model,
    image_path,
    confidence=DEFAULT_CONFIDENCE,
    device=None,
    target_class=DEFAULT_CLASS_NAME,
):
    """
    Run YOLO26 and return standardized 2D detections.

    Ultralytics performs its own confidence processing and NMS.
    This adapter does not apply an additional deduplication step.
    """

    image_path = Path(
        image_path
    )

    predict_args = {
        "source": str(image_path),
        "conf": float(confidence),
        "verbose": False,
    }

    if device is not None:
        predict_args["device"] = device

    result = model.predict(
        **predict_args
    )[0]

    if result.boxes is None:
        boxes_xyxy = np.empty(
            (0, 4),
            dtype=np.float32,
        )

        scores = np.empty(
            (0,),
            dtype=np.float32,
        )

        class_ids = np.empty(
            (0,),
            dtype=np.int64,
        )

    else:
        boxes_xyxy = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        class_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    standardized = standardize_yolo_detections(
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        class_ids=class_ids,
        class_names=model.names,
        target_class=target_class,
    )

    if hasattr(
        result,
        "orig_shape",
    ):
        image_height = int(
            result.orig_shape[0]
        )

        image_width = int(
            result.orig_shape[1]
        )

    else:
        image_height = None
        image_width = None

    standardized.update({
        "image_path": str(
            image_path
        ),

        "image_width": (
            image_width
        ),

        "image_height": (
            image_height
        ),

        "confidence": float(
            confidence
        ),

        "target_class": str(
            target_class
        ),
    })

    return standardized
