# CLOCs reduced-cloud validation parity run (29 September 2026)

## Why this run exists

The historical PointPillars full validation loaded KITTI `training/velodyne_reduced`
(camera 2 field-of-view points) and applied the pinned test pipeline. The first
CLOCs full validation loaded raw `training/velodyne` clouds. Replaying the
verified original predictions against the newer ground truth/evaluator reproduced
all nine historical AP40 values exactly. A three-frame reduced-cloud probe
reproduced historical PointPillars counts and scores exactly on frames 000001,
000004, and 000831; raw clouds did not.

This is a separate validation run. Keep the existing raw-cloud evidence intact.
The CLOCs checkpoint was trained from raw-cloud candidates, so any reduced-cloud
validation result also tests a preprocessing change at evaluation time.

## Source and inputs

Run with pinned MMDetection3D fe25f7a51d36e3702f961e198894580d83c4387b,
PyTorch 2.1.2+cu121, the same checkpoint/config, YOLO26x and CLOCs checkpoint.
The new `fusion/kitti_reduced_cloud.py` recreates the pinned
`tools/dataset_converters/kitti_converter.py` camera 2 reduction with
`box_np_ops.remove_outside_points`. The runner creates each reduced point
cloud transiently, then removes it. It checks retained PointPillars counts,
scores and geometry against the SHA-pinned historical prediction pickle before
writing that frame. A mismatch stops the run.

The saved provenance includes `point_cloud_mode=reduced_fov`. Use a *new*
output directory. The old raw-cloud prediction directory must not be reused.

## Smoke then full run

Create a fresh code ZIP from the current repository and attach it as a Kaggle
dataset along with the existing full-validation inputs, historical replay inputs,
model checkpoints and weights. Run with the pinned Python environment and
`PYTHONPATH` set to the mounted fresh code root. The commands below assume
those mounts and a still-present `/kaggle/working/mmdetection3d` checkout.

```bash
PY=/kaggle/working/pointpillars310/bin/python
CODE=/kaggle/input/datasets/abhishekjain18/clocs-reduced-validation-code-zip
DATA=/kaggle/input/datasets/abhishekjain18/clocs-kitti-validation-3769-tar-gz
HISTORY=/kaggle/input/datasets/abhishekjain18/clocs-historical-replay-inputs-zip/pred_instances_3d.pkl
CONFIG=/kaggle/working/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py
PP=/kaggle/input/datasets/abhishekjain18/pointpillars-kitti-car-pth/pointpillars_kitti_car.pth
YOLO=/kaggle/input/datasets/abhishekjain18/yolo26x-pt/yolo26x.pt
FUSION=/kaggle/input/datasets/abhishekjain18/clocs-val-smoke-kaggle-inputs-zip/clocs_fusion_best.pt
OUT=/kaggle/working/clocs_full_val_predictions_reduced

PYTHONPATH="$CODE" MPLBACKEND=Agg "$PY" -m scripts.run_clocs_full_validation \
  --dataset "$DATA/training" --manifest "$DATA/manifest.json" \
  --config "$CONFIG" --pointpillars-checkpoint "$PP" \
  --yolo-weights "$YOLO" --fusion-checkpoint "$FUSION" \
  --point-cloud-mode reduced_fov --historical-predictions "$HISTORY" \
  --output "$OUT" --max-new-frames 4
```

Inspect the four newly created files. Rerun the same command without
`--max-new-frames 4` to resume all 3,769 frames. The runner verifies and
reuses completed frames. Do not delete a partial output without inspecting it.

## Evaluate after complete inference

```bash
PYTHONPATH="$CODE" MPLBACKEND=Agg NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
  "$PY" -m scripts.evaluate_clocs_full_validation \
  --manifest "$DATA/manifest.json" --predictions "$OUT" \
  --point-cloud-mode reduced_fov \
  --output /kaggle/working/clocs_full_val_reduced_ap40.json
```

The evaluator applies the historical AP40 baseline gate (maximum allowed
absolute difference 0.05 AP points across nine strict Car metrics) before
computing fusion AP. Record the output and prediction digests before interpreting
the new result. This is KITTI validation, not the official test server.
