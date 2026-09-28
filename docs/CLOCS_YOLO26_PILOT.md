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
and dense export provenance. The subsequent Mac execution is recorded below. Kaggle project compatibility
and training remain pending. No trained CLOCs
accuracy or generalisation result is claimed by this preparation step.

## Frame 000149 execution receipt

Source: user-supplied Mac terminal output on 28 September 2026, after pulling
`37e7a51`. The full Mac suite passed 112 tests. The frame builder completed with
camera `yolo26x`, Ultralytics 8.4.132, MPS and confidence threshold 0.25.

- Car GT: 10; YOLO26x Car detections: 9.
- Candidates: 107136; feature rows: 158691.
- Positive / negative / ignored: 179 / 106798 / 159.
- Best IoU maximum: 0.9291334748268127; nonzero: 5137.
- Feature-row file SHA-256: `fa1d2b001fd886bf243149c9c85bf2b9a045357533e7bebe7fda3dedd271b863`.
- Training-example content SHA-256: `01de9ddccbde0ebe99bd93f03c9e16728da34be56937b00b656d79d467cc8c9a`.

The unchanged labels agree with the existing YOLOv8 control. There are 147 fewer
feature rows than the YOLOv8 example (158838); this is a camera-association
change, not evidence of improved detector accuracy. The original small audit
is `results/clocs_yolo26_frame_000149_audit.json` on the Mac, awaiting its own
commit at the time of this receipt. Large files remain in the separate
`yolo26x` directory outside Git. This receipt records reported execution;
it is not an independent rerun of the local artifacts.
