#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import load_image
from falcon_perception.mlx.batch_inference import (
    BatchInferenceEngine,
    process_batch_and_generate,
)


def pair_bbox_entries(raw):
    """
    Falcon returns bbox information as separate coordinate/size entries.
    Convert them into complete {x, y, h, w} dictionaries.
    """
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


def normalized_to_pixels(bbox, width, height):
    """
    Convert Falcon normalized center-format bbox:
        cx, cy, w, h

    into pixel-format:
        x1, y1, x2, y2
    """

    cx = float(bbox["x"])
    cy = float(bbox["y"])
    bw = float(bbox["w"])
    bh = float(bbox["h"])

    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    return x1, y1, x2, y2


def run_falcon(
    image_path,
    query,
    dtype="float16",
    max_new_tokens=200,
):
    image_path = Path(image_path)

    print("Loading Falcon Perception MLX model...")

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype=dtype,
        backend="mlx",
    )

    image = load_image(
        str(image_path)
    ).convert("RGB")

    width, height = image.size

    prompt = build_prompt_for_task(
        query,
        "detection",
    )

    batch = process_batch_and_generate(
        tokenizer,
        [(image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=256,
        max_dimension=1024,
    )

    engine = BatchInferenceEngine(
        model,
        tokenizer,
    )

    print("Running Falcon detection...")

    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        task="detection",
    )

    aux = aux_outputs[0]

    bboxes = pair_bbox_entries(
        aux.bboxes_raw
    )

    detections = []

    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = normalized_to_pixels(
            bbox,
            width,
            height,
        )

        detections.append({
            "detection_id": index,
            "class": query,
            "query": query,
            "model": "Falcon-Perception",
            "normalized_bbox": {
                "cx": float(bbox["x"]),
                "cy": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            },
            "pixel_bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    return {
        "frame": image_path.stem,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "query": query,
        "model": "Falcon-Perception",
        "backend": "MLX",
        "num_detections": len(detections),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Falcon Perception detection adapter"
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--query",
        default="car",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    args = parser.parse_args()

    result = run_falcon(
        image_path=args.image,
        query=args.query,
        dtype=args.dtype,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        f"Detections: {result['num_detections']}"
    )

    print(
        f"Saved JSON: {output_path}"
    )

    for detection in result["detections"]:
        box = detection["pixel_bbox"]

        print(
            f"[{detection['detection_id']:02d}] "
            f"x1={box['x1']:.1f} "
            f"y1={box['y1']:.1f} "
            f"x2={box['x2']:.1f} "
            f"y2={box['y2']:.1f}"
        )


if __name__ == "__main__":
    main()
