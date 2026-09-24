#!/usr/bin/env python3

import unittest

import numpy as np

from adapters.yolo26_adapter import (
    extract_yolo26_detections,
    standardize_yolo_detections,
)


class FakeTensor:

    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:

    def __init__(
        self,
        xyxy,
        conf,
        cls,
    ):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:

    def __init__(
        self,
        boxes,
        orig_shape=(375, 1242),
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape


class FakeModel:

    def __init__(self):
        self.names = {
            0: "person",
            2: "car",
        }

        self.last_predict_args = None

    def predict(self, **kwargs):
        self.last_predict_args = kwargs

        boxes = FakeBoxes(
            xyxy=[
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
            ],
            conf=[
                0.90,
                0.70,
                0.60,
            ],
            cls=[
                2,
                0,
                2,
            ],
        )

        return [
            FakeResult(
                boxes=boxes,
            )
        ]


class TestYOLO26Adapter(unittest.TestCase):

    def test_standardize_filters_car_and_preserves_order(self):
        result = standardize_yolo_detections(
            boxes_xyxy=np.array([
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
            ], dtype=np.float32),

            scores=np.array([
                0.90,
                0.70,
                0.60,
            ], dtype=np.float32),

            class_ids=np.array([
                2,
                0,
                2,
            ], dtype=np.int64),

            class_names={
                0: "person",
                2: "car",
            },

            target_class="car",
        )

        np.testing.assert_array_equal(
            result["boxes_2d"],
            np.array([
                [10, 20, 30, 40],
                [90, 100, 110, 120],
            ], dtype=np.float32),
        )

        np.testing.assert_allclose(
            result["scores_2d"],
            np.array([
                0.90,
                0.60,
            ], dtype=np.float32),
        )

        self.assertEqual(
            result["detection_index"].tolist(),
            [0, 2],
        )

        self.assertEqual(
            result["num_detections"],
            2,
        )

    def test_class_matching_is_case_insensitive(self):
        result = standardize_yolo_detections(
            boxes_xyxy=np.array([
                [1, 2, 3, 4],
            ], dtype=np.float32),

            scores=np.array([
                0.8,
            ], dtype=np.float32),

            class_ids=np.array([
                2,
            ], dtype=np.int64),

            class_names=[
                "person",
                "bicycle",
                "car",
            ],

            target_class="Car",
        )

        self.assertEqual(
            result["num_detections"],
            1,
        )

    def test_empty_detections(self):
        result = standardize_yolo_detections(
            boxes_xyxy=np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            scores=np.empty(
                (0,),
                dtype=np.float32,
            ),
            class_ids=np.empty(
                (0,),
                dtype=np.int64,
            ),
            class_names={
                2: "car",
            },
        )

        self.assertEqual(
            result["boxes_2d"].shape,
            (0, 4),
        )

        self.assertEqual(
            result["scores_2d"].shape,
            (0,),
        )

        self.assertEqual(
            result["num_detections"],
            0,
        )

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(
            ValueError
        ):
            standardize_yolo_detections(
                boxes_xyxy=np.array([
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ], dtype=np.float32),

                scores=np.array([
                    0.9,
                ], dtype=np.float32),

                class_ids=np.array([
                    2,
                    2,
                ], dtype=np.int64),

                class_names={
                    2: "car",
                },
            )

    def test_extract_yolo26_detections(self):
        model = FakeModel()

        result = extract_yolo26_detections(
            model=model,
            image_path="/tmp/000001.png",
            confidence=0.25,
            device="mps",
            target_class="car",
        )

        self.assertEqual(
            result["num_detections"],
            2,
        )

        self.assertEqual(
            result["detection_index"].tolist(),
            [0, 2],
        )

        self.assertEqual(
            result["image_width"],
            1242,
        )

        self.assertEqual(
            result["image_height"],
            375,
        )

        self.assertEqual(
            result["confidence"],
            0.25,
        )

        self.assertEqual(
            result["target_class"],
            "car",
        )

        self.assertEqual(
            model.last_predict_args["source"],
            "/tmp/000001.png",
        )

        self.assertEqual(
            model.last_predict_args["device"],
            "mps",
        )

        self.assertEqual(
            model.last_predict_args["conf"],
            0.25,
        )

        self.assertFalse(
            model.last_predict_args["verbose"]
        )


if __name__ == "__main__":
    unittest.main()
