# YOLO26x CLOCs pilot — 28 September 2026

The planned main KITTI fusion experiment uses YOLO26x with frozen PointPillars.
The existing 16-frame YOLOv8x pilot remains valid engineering evidence under its
original identity. It must not be relabelled as YOLO26x or overwritten.

`scripts/build_clocs_yolo_train_frame.py` accepts `--camera-model yolo26x` and
validates the selected camera weight hash before inference. The frozen local
YOLO26x hash reported on the Mac is
`9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.
This is project artifact provenance, not independent upstream authenticity
verification. The Mac reported Ultralytics 8.4.132.

New YOLO26 audits use `clocs_real_train_frame_audit_v2`, `yolo26x_car_detections`
and `yolo26x_weights`, and record camera identity, device, confidence threshold
and Ultralytics version. Feature-row archives also identify the camera and hash.
The existing YOLOv8 default keeps its v1 field names for the old batch consumer.
All output paths must be new; existing artifacts cause an error before inference.

Use the existing 000149 dense candidates for the first YOLO26 example. Store
new rows/targets under `model_outputs/clocs_stage5_checkpoints/yolo26x/` and the
small audit as `results/clocs_yolo26_frame_000149_audit.json`. The unchanged GT
and candidates should retain 107136 candidates and positive/negative/ignored
counts 179/106798/159. Camera detection and feature-row counts can differ.
Compare actual output with those control counts before expanding the pilot.

The untracked 128-frame Mac manifest reports `INPUTS_VERIFIED_EXPORT_PENDING`;
only 16 dense exports were confirmed present on 28 September. A manifest does
not establish completed inference or training.

Four focused tests passed in the development workspace: camera weight identity
and tamper rejection, existing artifact protection, train-only split validation,
and dense export provenance. Real YOLO26 inference on this training frame and
Kaggle project compatibility remain pending user execution. No trained CLOCs
accuracy or generalisation result is claimed by this preparation step.
